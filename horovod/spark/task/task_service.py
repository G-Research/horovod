# Copyright 2019 Uber Technologies, Inc. All Rights Reserved.
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

from distutils.version import LooseVersion

import os
import pyspark
import time

from horovod.run.common.util import codec, secret, timeout
from horovod.run.common.service import task_service


class ResourcesRequest(object):
    """Request Spark resources info for this task."""


class ResourcesResponse(object):
    def __init__(self, resources):
        self.resources = resources
        """Dictionary containing resource info."""


class GetTaskToTaskAddressesRequest(object):
    def __init__(self, task_index, all_task_addresses):
        self.task_index = task_index
        """Task index of other task service."""

        self.all_task_addresses = all_task_addresses
        """Map of interface to list of (ip, port) pairs of other task service."""


class GetTaskToTaskAddressesResponse(object):
    def __init__(self, task_addresses_for_task):
        self.task_addresses_for_task = task_addresses_for_task
        """Map of interface to list of (ip, port) pairs."""


class SparkTaskService(task_service.BasicTaskService):
    NAME_FORMAT = 'task service #%d'

    def __init__(self, index, key, nics, minimum_command_lifetime_s, verbose=0):
        # on a Spark cluster we need our train function to see the Spark worker environment
        # this includes PYTHONPATH, HADOOP_TOKEN_FILE_LOCATION and _HOROVOD_SECRET_KEY
        env = os.environ.copy()

        # we inject the secret key here
        env[secret.HOROVOD_SECRET_KEY] = codec.dumps_base64(key)

        # we also need to provide the current working dir to mpirun_exec_fn.py
        env['HOROVOD_SPARK_WORK_DIR'] = os.getcwd()

        super(SparkTaskService, self).__init__(SparkTaskService.NAME_FORMAT % index,
                                               key, nics, env, verbose)
        self._key = key
        self._minimum_command_lifetime_s = minimum_command_lifetime_s
        self._minimum_command_lifetime = None

    def _run_command(self, command, env, event):
        super(SparkTaskService, self)._run_command(command, env, event)

        if self._minimum_command_lifetime_s is not None:
            self._minimum_command_lifetime = timeout.Timeout(self._minimum_command_lifetime_s,
                                                             message='Just measuring runtime')

    def _handle(self, req, client_address):
        if isinstance(req, ResourcesRequest):
            return ResourcesResponse(self._get_resources())

        if isinstance(req, GetTaskToTaskAddressesRequest):
            next_task_index = req.task_index
            next_task_addresses = req.all_task_addresses
            # We request interface matching to weed out all the NAT'ed interfaces.
            next_task_client = \
                SparkTaskClient(next_task_index, next_task_addresses,
                                self._key, self._verbose,
                                match_intf=True)
            return GetTaskToTaskAddressesResponse(next_task_client.addresses())

        return super(SparkTaskService, self)._handle(req, client_address)

    def _get_resources(self):
        if LooseVersion(pyspark.__version__) >= LooseVersion('3.0.0'):
            from pyspark import TaskContext
            return TaskContext.get().resources()
        return dict()

    def wait_for_command_termination(self):
        """
        Waits for command termination. Ensures this method takes at least
        self._minimum_command_lifetime_s seconds to return after command started.
        """
        super(SparkTaskService, self).wait_for_command_termination()

        # command terminated, make sure this method takes at least
        # self._minimum_command_lifetime_s seconds after command started
        # the client that started the command needs some time to connect again
        # to wait for the result (see horovod.spark.driver.rsh).
        if self._minimum_command_lifetime is not None:
            time.sleep(self._minimum_command_lifetime.remaining())


class SparkTaskClient(task_service.BasicTaskClient):

    def __init__(self, index, task_addresses, key, verbose, match_intf=False):
        super(SparkTaskClient, self).__init__(SparkTaskService.NAME_FORMAT % index,
                                              task_addresses, key, verbose,
                                              match_intf=match_intf)

    def resources(self):
        resp = self._send(ResourcesRequest())
        return resp.resources

    def get_task_addresses_for_task(self, task_index, all_task_addresses):
        resp = self._send(GetTaskToTaskAddressesRequest(task_index, all_task_addresses))
        return resp.task_addresses_for_task
