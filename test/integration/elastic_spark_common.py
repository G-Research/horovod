# Copyright 2020 Uber Technologies, Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from __future__ import absolute_import

import contextlib
import json
import logging
import os
import re
import sys
import threading
import time
import warnings

import mock
import pytest
import unittest

from horovod.run.util.threads import in_thread
from horovod.run.common.util import safe_shell_exec, tiny_shell_exec
from horovod.spark import run_elastic

sys.path.append(os.path.join(os.path.dirname(__file__), os.pardir))

from common import temppath


@contextlib.contextmanager
def spark_cluster(logfile, discovery_schedule, hosts, extra_conf=None):
    from pyspark import SparkConf
    from pyspark.sql import SparkSession

    cluster = SparkClusterController(logfile, discovery_schedule, hosts, 1)
    cluster.start()

    conf = SparkConf().setAppName('elastic spark tests').setMaster(cluster.master_url())
    conf = conf.setAll([
        # pyspark-shell JVM will OOM even with 1g when all tests run in one process
        # SparkContext and pyspark-shell jvm gets reused even though we do SparkSession.stop()
        # pyspark-shell JVM memory footprint increases from test to test
        # when run with pytest --forked, set SPARK_DRIVER_MEM=512m env
        ('spark.driver.memory', os.environ.get('SPARK_DRIVER_MEM', '1500m')),
        # the minimum executor memory we can set
        ('spark.executor.memory', '512m'),
        ('spark.ui.showConsoleProgress', 'false'),
        ('spark.test.home', os.environ.get('SPARK_HOME')),
        ('spark.locality.wait', '0'),
        # never let Spark kill the Horovod app, Horovod has its own mechanisms
        ('spark.task.maxFailures', '2147483647'),
        ('spark.blacklist.enabled', 'true'),
        # blacklist the executor on first failure, this syncs Spark's blacklisting with Horovod's
        ('spark.blacklist.stage.maxFailedTasksPerExecutor', '1'),
        # never let Spark blacklist a node on task failure, Horovod has its own mechanisms
        ('spark.blacklist.task.maxTaskAttemptsPerNode', '2147483646'),
        ('spark.blacklist.stage.maxFailedExecutorsPerNode', '2147483646'),
    ])
    conf.setAll(extra_conf.items() if extra_conf is not None else {})

    session = SparkSession \
        .builder \
        .config(conf=conf) \
        .getOrCreate()

    try:
        yield session
    finally:
        session.stop()
        cluster.shutdown()


class SparkClusterController(object):
    """
    Controls a stand-alone Spark cluster for integration tests.
    Tests can run forked, but not in parallel.
    Multiple controller instances would restart each other's masters.
    """
    def __init__(self, logfile, discovery_schedule, hosts, master_instance):
        if hosts and discovery_schedule or (not hosts and not discovery_schedule):
            raise ValueError('either discovery schedule or hosts must be given, not both')

        if 'SPARK_HOME' not in os.environ:
            raise RuntimeError('SPARK_HOME not set, it has to point to your Spark installation')

        self._spark_home = os.environ['SPARK_HOME']
        self._logfile = logfile
        self._discovery_schedule = dict(discovery_schedule) if discovery_schedule else {}
        self._hosts = hosts.split(',') if hosts else None
        self._next_worker_instance = master_instance
        self._host_worker = {}
        self._workers = []
        self._shutdown = threading.Event()
        self._logfile_monitor = None

        self._master_instance = master_instance
        self._master_host = 'localhost'
        self._master_url = None
        self._log_re = re.compile('^starting org.apache.spark.deploy.master.Master, logging to (.+)$')
        self._url_re = re.compile('^.+ Master: Starting Spark master at (.+)[\r\n]+')

    def master_url(self):
        return self._master_url

    def start(self):
        logging.info('starting Spark cluster')
        if not self.start_master():
            self.stop_master()
            if not self.start_master():
                raise RuntimeError('could not start master')

        self._logfile_monitor = in_thread(self._monitor_logfile)
        if self._hosts:
            self.provide_hosts(self._hosts)

    def start_master(self):
        logging.info('starting Spark master %s', self._master_instance)
        res = self._execute('env SPARK_MASTER_OPTS=-Dspark.deploy.maxExecutorRetries=-1 '
                            '{spark_home}/sbin/spark-daemon.sh '
                            'start org.apache.spark.deploy.master.Master {instance} '
                            '--host {host} '
                            '--port 0'.format(spark_home=self._spark_home,
                                              instance=self._master_instance,
                                              host=self._master_host))
        if not res or res[1]:
            return False

        self._master_url = self._get_master_url(self._get_log_file(res[0]))
        return self._master_url is not None

    def _get_log_file(self, line):
        log = self._log_re.match(line)
        if log:
            return log.group(1)
        return None

    def _get_master_url(self, log):
        if not log:
            return None

        timer = time.time()
        while time.time() - timer < 2.0:
            with open(log, 'r') as f:
                lines = f.readlines()

            for line in lines:
                url = self._url_re.match(line)
                if url:
                    logging.info('master url is {}'.format(url.group(1)))
                    return url.group(1)

            time.sleep(0.1)
        return None

    def start_worker(self, instance, cores):
        logging.info('starting Spark worker %d with %d cores', instance, cores)
        res = self._execute('env PYTHONPATH={pythonpath} '
                            '{spark_home}/sbin/spark-daemon.sh '
                            'start org.apache.spark.deploy.worker.Worker {instance} '
                            '-c {cores} '
                            '{master_url}'.format(pythonpath=os.pathsep.join(sys.path),
                                                  spark_home=self._spark_home,
                                                  instance=instance,
                                                  cores=cores,
                                                  master_url=self.master_url()))
        if not res or res[1]:
            return False

        self._workers.append(instance)
        return True

    def start_or_restart_worker(self, instance, cores):
        if not self.start_worker(instance, cores):
            self.stop_worker(instance)
            if not self.start_worker(instance, cores):
                raise RuntimeError('could not start worker {}'.format(instance))

    def provide_hosts(self, hosts):
        """Makes the cluster provide the given hosts only.

        Any host currently provided that is not in the given hosts will be shut down.
        This does not allow for changes in the number of slots.
        """
        logging.debug('make Spark cluster provide hosts %s', hosts)

        # shut down missing works first
        for host in self._host_worker.copy():
            if host not in hosts:
                in_thread(self.stop_worker, args=(self._host_worker[host],))
                del self._host_worker[host]

        # start new workers
        for host in hosts:
            if host not in self._host_worker:
                cores = int(host.split(':', 1)[1])
                instance = self._next_worker_instance
                in_thread(self.start_or_restart_worker, args=(instance, cores))
                self._host_worker[host] = instance
                self._next_worker_instance += 1

    def shutdown(self):
        logging.info('shutting down Spark cluster')
        self._shutdown.set()
        self.stop_workers()
        self.stop_master()
        self._logfile_monitor.join()

    def stop_workers(self):
        for instance in self._workers.copy():
            in_thread(self.stop_worker(instance), daemon=False)

    def stop_worker(self, instance):
        logging.info('stopping Spark worker %s', instance)
        self._execute('{spark_home}/sbin/spark-daemon.sh '
                      'stop org.apache.spark.deploy.worker.Worker {instance}'
                      .format(spark_home=self._spark_home,
                              instance=instance))
        if instance in self._workers:
            self._workers.remove(instance)

    def stop_master(self):
        logging.info('stopping Spark master')
        self._execute('{spark_home}/sbin/spark-daemon.sh '
                      'stop org.apache.spark.deploy.master.Master {instance}'
                      .format(spark_home=self._spark_home,
                              instance=self._master_instance))

    def _execute(self, command):
        logging.debug('executing %s', command)
        res = tiny_shell_exec.execute(command)

        if res:
            if res[1]:
                logging.warning('command failed: {}'.format(res[0].strip()))
            else:
                logging.debug('command succeeded: {}'.format(res[0].strip()))
        else:
            logging.error('command failed: {}'.format(command))

        return res

    def _monitor_logfile(self):
        logging.info('monitoring logfile %s', self._logfile)

        last_epoch = -1
        while not self._shutdown.is_set():
            if os.path.exists(self._logfile):
                with open(self._logfile, 'r') as f:
                    lines = f.readlines()
                epoch = len(lines)
            else:
                epoch = 0

            if epoch != last_epoch:
                logging.debug('epoch %d -> %d', last_epoch, epoch)
                if epoch in self._discovery_schedule:
                    self.provide_hosts(self._discovery_schedule[epoch])
                    del self._discovery_schedule[epoch]
                elif None in self._discovery_schedule:
                    self.provide_hosts(self._discovery_schedule[None])
                    if len(self._discovery_schedule) == 1:
                        del self._discovery_schedule[None]

            last_epoch = epoch
            self._shutdown.wait(0.1)


class BaseElasticSparkTests(unittest.TestCase):
    """
    Tests Elastic Horovod on Spark with against a stand-alone Spark cluster.

    The cluster is brought up and controlled by the unit test.
    Tests can run forked but not in parallel.
    Running tests in parallel would put CPU under pressure.

    Running tests forked will reduce memory footprint from 1.5GB for the master to 512MB.
    Each executor (slot) needs 512GB memory. There are at most 6 executors used in these tests.
    Set SPARK_DRIVER_MEM=512m env when run forked to reduce driver memory usage.
    """

    # do not run these tests under BaseElasticSparkTests but as tests of subclasses.
    __test__ = False

    def __init__(self, training_script, *args, **kwargs):
        self._training_script = training_script
        super(BaseElasticSparkTests, self).__init__(*args, **kwargs)
        warnings.simplefilter('module')
        logging.getLogger('py4j.java_gateway').setLevel(logging.INFO)

    @staticmethod
    def _exec(cmd):
        exit_code = safe_shell_exec.execute(cmd)
        if exit_code is None or exit_code != 0:
            raise RuntimeError('executed command returned non-zero exit code: {}'.format(exit_code))

    def _run(self, discovery_schedule=None, exit_schedule=None, hosts=None, epochs=None,
             np=2, min_np=None, max_np=None, extra_conf=None):
        with temppath() as logfile:
            with spark_cluster(logfile=logfile, discovery_schedule=discovery_schedule,
                               hosts=hosts, extra_conf=extra_conf):
                command = [sys.executable, self._training_script, '--logfile', logfile]
                if discovery_schedule:
                    command += ['--discovery-schedule', "'{}'".format(json.dumps(discovery_schedule)),
                                '--discovery-wait', '10']
                if exit_schedule:
                    command += ['--exit-schedule', "'{}'".format(json.dumps(exit_schedule))]
                if epochs:
                    command += ['--epochs', str(epochs)]

                cmd = ' '.join(command)
                run_elastic(self._exec, (cmd,), env={'HOROVOD_LOG_LEVEL': 'DEBUG'},
                            num_proc=np, min_np=min_np, max_np=max_np,
                            start_timeout=10, elastic_timeout=10, verbose=2)

                with open(logfile, 'r') as f:
                    lines = f.readlines()

                print('logfile:')
                for line in lines:
                    print(line)

                return [json.loads(line) for line in lines]

    @mock.patch('horovod.run.elastic.driver.DISCOVER_HOSTS_FREQUENCY_SECS', 0.01)
    def test_happy_run(self):
        hosts = 'host-1:1,host-2:2,host-3:3'

        epochs = 10
        self.assertGreater(epochs, 0, 'test should not be trivial')
        results = self._run(hosts=hosts, np=5, min_np=5, max_np=5, epochs=epochs)

        self.assertEqual(epochs, len(results))

        host = results[0]['hostname']
        for epoch in range(epochs):
            self.assertEqual(0, results[epoch]['rank'])
            self.assertEqual(0, results[epoch]['start_rank'])
            self.assertEqual(5, results[epoch]['size'])
            self.assertEqual(1, results[epoch]['rendezvous'])
            self.assertEqual(host, results[epoch]['hostname'])
            self.assertEqual(10, results[epoch]['batch'])
            self.assertEqual(10 + epoch * 11, results[epoch]['commits'])

    @mock.patch('horovod.run.elastic.driver.DISCOVER_HOSTS_FREQUENCY_SECS', 0.01)
    def test_fault_tolerance_no_spark_black_list(self):
        hosts = 'host-1:1,host-2:1'

        exit_schedule = {
            str((1, 0)): [0],
        }

        results = self._run(hosts=hosts, exit_schedule=exit_schedule, np=2,
                            extra_conf={'spark.blacklist.enabled': 'false'})

        self.assertEqual(3, len(results))

        self.assertEqual(0, results[0]['start_rank'])
        self.assertEqual(2, results[0]['size'])
        self.assertEqual(1, results[0]['rendezvous'])

        self.assertEqual(1, results[1]['start_rank'])
        self.assertEqual(2, results[1]['size'])
        self.assertEqual(2, results[1]['rendezvous'])

        self.assertEqual(1, results[2]['start_rank'])
        self.assertEqual(2, results[2]['size'])
        self.assertEqual(2, results[2]['rendezvous'])

    @mock.patch('horovod.run.elastic.driver.DISCOVER_HOSTS_FREQUENCY_SECS', 0.01)
    def test_fault_tolerance_spark_black_list(self):
        """
        Same as test_fault_tolerance_no_spark_black_list except Spark blacklists the executor
        that has the failing task, so that there are not enough executors available after the
        exception. Then, horovod will timeout waiting for np=2 cores.
        """
        hosts = 'host-1:1,host-2:1'

        exit_schedule = {
            str((1, 0)): [0],
        }

        message = 'Horovod detected that one or more processes exited with non-zero status'
        with pytest.raises(RuntimeError, match=message):
            self._run(hosts=hosts, exit_schedule=exit_schedule, np=2)

    @mock.patch('horovod.run.elastic.driver.DISCOVER_HOSTS_FREQUENCY_SECS', 0.01)
    def test_fault_tolerance_hosts_added_and_removed(self):
        discovery_schedule = [
            (0, ['host-1:1', 'host-2:1']),
            (1, ['host-2:1', 'host-3:1']),
            (None, ['host-3:1', 'host-4:1']),
        ]

        results = self._run(discovery_schedule=discovery_schedule)

        self.assertEqual(3, len(results))

        self.assertEqual(0, results[0]['start_rank'])
        self.assertEqual(2, results[0]['size'])
        self.assertEqual(1, results[0]['rendezvous'])

        # either host for rank 0 got removed, or host for rank 1
        if results[1]['start_rank'] == results[0]['start_rank']:
            # host of rank 1 got removed in epoch 1, rank 0 unchanged
            self.assertEqual(0, results[1]['start_rank'])
            self.assertEqual(2, results[1]['size'])
            self.assertEqual(results[0]['hostname'], results[1]['hostname'])
            self.assertEqual(2, results[1]['rendezvous'])

            # host of rank 0 got removed in epoch 2
            self.assertEqual(1, results[2]['start_rank'])
            self.assertEqual(2, results[2]['size'])
            self.assertNotEqual(results[1]['hostname'], results[2]['hostname'])
            self.assertEqual(3, results[2]['rendezvous'])
        else:
            # host of rank 0 got removed in epoch 1
            self.assertEqual(1, results[1]['start_rank'])
            self.assertEqual(2, results[1]['size'])
            self.assertNotEqual(results[0]['hostname'], results[1]['hostname'])
            self.assertEqual(2, results[1]['rendezvous'])

            # host of rank 1 got removed in epoch 2, rank 0 unchanged
            self.assertEqual(1, results[2]['start_rank'])
            self.assertEqual(2, results[2]['size'])
            self.assertEqual(results[1]['hostname'], results[2]['hostname'])
            self.assertEqual(3, results[2]['rendezvous'])

    @mock.patch('horovod.run.elastic.driver.DISCOVER_HOSTS_FREQUENCY_SECS', 0.01)
    def test_fault_tolerance_exception_single_rank(self):
        hosts = 'host-1:2,host-2:2'

        exit_schedule = {
            str((1, 0)): [0],
        }

        results = self._run(hosts=hosts, exit_schedule=exit_schedule, np=2, min_np=2, max_np=2)

        self.assertEqual(3, len(results))

        self.assertEqual(0, results[0]['start_rank'])
        self.assertEqual(2, results[0]['size'])
        self.assertEqual(1, results[0]['rendezvous'])

        self.assertEqual(1, results[1]['start_rank'])
        self.assertEqual(2, results[1]['size'])
        self.assertEqual(2, results[1]['rendezvous'])

        self.assertEqual(1, results[2]['start_rank'])
        self.assertEqual(2, results[2]['size'])
        self.assertEqual(2, results[2]['rendezvous'])

    @mock.patch('horovod.run.elastic.driver.DISCOVER_HOSTS_FREQUENCY_SECS', 0.01)
    def test_fault_tolerance_exception_all_ranks(self):
        hosts = 'localhost:2'

        exit_schedule = {
            str((1, 0)): [0, 1],
        }

        message = 'Horovod detected that one or more processes exited with non-zero status'
        with pytest.raises(RuntimeError, match=message):
            self._run(hosts=hosts, exit_schedule=exit_schedule)

    @mock.patch('horovod.run.elastic.driver.DISCOVER_HOSTS_FREQUENCY_SECS', 0.01)
    def test_fault_tolerance_exception_with_min_hosts_timeout(self):
        hosts = 'host-1:1,host-2:1'

        exit_schedule = {
            str((1, 0)): [0],
        }

        message = 'Horovod detected that one or more processes exited with non-zero status'
        with pytest.raises(RuntimeError, match=message):
            self._run(hosts=hosts, exit_schedule=exit_schedule, np=2)

    @mock.patch('horovod.run.elastic.driver.DISCOVER_HOSTS_FREQUENCY_SECS', 0.01)
    def test_fault_tolerance_all_hosts_lost(self):
        discovery_schedule = [
            (0, ['node-1:1', 'node-2:1']),
            (None, []),
        ]

        message = 'Horovod detected that one or more processes exited with non-zero status'
        with pytest.raises(RuntimeError, match=message):
            self._run(discovery_schedule=discovery_schedule)

    @mock.patch('horovod.run.elastic.driver.DISCOVER_HOSTS_FREQUENCY_SECS', 0.01)
    @mock.patch('horovod.run.gloo_run._get_min_start_hosts', return_value=1)
    def test_auto_scale_up(self, mock_get_min_start_hosts):
        # TODO: can be removed once test_auto_scale_up_and_down works
        discovery_schedule = [
            (0, ['host-1:1']),
            (1, ['host-1:1', 'host-2:1']),
            (None, ['host-1:1', 'host-2:1', 'host-3:1']),
        ]

        results = self._run(discovery_schedule=discovery_schedule, np=1, min_np=1, max_np=5)

        self.assertEqual(3, len(results))

        self.assertEqual(0, results[0]['start_rank'])
        self.assertEqual(1, results[0]['size'])
        self.assertEqual(1, results[0]['rendezvous'])

        self.assertEqual(0, results[1]['start_rank'])
        self.assertEqual(2, results[1]['size'])
        self.assertEqual(2, results[1]['rendezvous'])
        self.assertEqual(results[0]['hostname'], results[1]['hostname'])

        self.assertEqual(0, results[2]['start_rank'])
        self.assertEqual(3, results[2]['size'])
        self.assertEqual(3, results[2]['rendezvous'])
        self.assertEqual(results[0]['hostname'], results[2]['hostname'])

    @mock.patch('horovod.run.elastic.driver.DISCOVER_HOSTS_FREQUENCY_SECS', 0.01)
    def test_auto_scale_down_by_discovery(self):
        discovery_schedule = [
            (0, ['host-1:1', 'host-2:1', 'host-3:1', 'host-4:1']),
            (1, ['host-2:1', 'host-3:1', 'host-4:1']),
            (2, ['host-2:1', 'host-3:1']),
            (None, ['host-3:1']),
        ]

        results = self._run(discovery_schedule=discovery_schedule, np=4, min_np=1, max_np=5, epochs=4)

        self.assertEqual(4, len(results))

        self.assertEqual(0, results[0]['start_rank'])
        self.assertEqual(4, results[0]['size'])
        self.assertEqual(1, results[0]['rendezvous'])

        self.assertEqual(3, results[1]['size'])
        self.assertEqual(2, results[1]['rendezvous'])

        self.assertEqual(2, results[2]['size'])
        self.assertEqual(3, results[2]['rendezvous'])

        self.assertEqual(1, results[3]['size'])
        self.assertEqual(4, results[3]['rendezvous'])

    @mock.patch('horovod.run.elastic.driver.DISCOVER_HOSTS_FREQUENCY_SECS', 0.01)
    def test_auto_scale_down_by_exception(self):
        hosts = 'host-1:1,host-2:1,host-3:1,host-4:1'

        exit_schedule = {
            str((1, 0)): [0],
            str((2, 0)): [1],
        }

        results = self._run(hosts=hosts, exit_schedule=exit_schedule, np=4, min_np=1)

        self.assertEqual(3, len(results))

        self.assertEqual(0, results[0]['start_rank'])
        self.assertEqual(4, results[0]['size'])
        self.assertEqual(1, results[0]['rendezvous'])

        self.assertEqual(1, results[1]['start_rank'])
        self.assertEqual(3, results[1]['size'])
        self.assertEqual(2, results[1]['rendezvous'])
        self.assertNotEqual(results[0]['hostname'], results[1]['hostname'])

        self.assertEqual(2, results[2]['start_rank'])
        self.assertEqual(2, results[2]['size'])
        self.assertEqual(3, results[2]['rendezvous'])
        self.assertNotEqual(results[0]['hostname'], results[2]['hostname'])
        self.assertNotEqual(results[1]['hostname'], results[2]['hostname'])

    @mock.patch('horovod.run.elastic.driver.DISCOVER_HOSTS_FREQUENCY_SECS', 0.01)
    def test_auto_scale_spark_black_list_multi_core_executor(self):
        """
        With spark.blacklist.stage.maxFailedTasksPerExecutor=1, a failed task causes Spark to
        black list the executor. Other running tasks will not be killed. Horovod black listing
        can then un-schedule workers on the same executor and terminate them gracefully.
        TODO: need to remove index and time from host hash for elastic horovod blacklisting
        """
        hosts = 'host-1:2,host-2:2'

        exit_schedule = {
            str((1, 0)): [1],
        }

        results = self._run(hosts=hosts, exit_schedule=exit_schedule, np=4, min_np=1)

        self.assertEqual(3, len(results))

        self.assertEqual(0, results[0]['start_rank'])
        self.assertEqual(4, results[0]['size'])
        self.assertEqual(1, results[0]['rendezvous'])

        self.assertEqual(0, results[1]['start_rank'])
        self.assertEqual(3, results[1]['size'])
        self.assertEqual(2, results[1]['rendezvous'])

        self.assertEqual(0, results[2]['start_rank'])
        self.assertEqual(3, results[2]['size'])
        self.assertEqual(2, results[2]['rendezvous'])
