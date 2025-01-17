# Copyright (c) 2022 Intel Corporation

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

'''
Wrappers around the distributed backends and helper functions
for doing collective communications
'''

from typing import ClassVar, List, cast, Optional, Callable, Any
import queue
import threading
import logging
import time
import os
import ifaddr  # type: ignore
import torch
import torch.distributed as dist
from torch import tensor
from .config import Config
from .common_tuples import SocketInfo

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())
logger.setLevel(logging.DEBUG)


def get_socket() -> SocketInfo:
    """
    Gets the socket on the current host. If preffered socket is not specified using SAR_SOCKET_NAME
    environment variable, the function returns the first available socket from `ifaddr.get_adapters()`
    
    :returns: Preffered or the first available socket
    """
    adaps = ifaddr.get_adapters()
    preferred_socket = os.environ.get("SAR_SOCKET_NAME")
    if preferred_socket is not None:
        adaps = list(filter(lambda x: x.nice_name == preferred_socket, adaps))
        if not adaps:
            raise ValueError(f'Socket with given name: "{preferred_socket}" was not found.')
    else:
        adaps = list(filter(lambda x: x.nice_name != "lo", adaps))
    return SocketInfo(adaps[0].nice_name, adaps[0].ips[0].ip)


def get_ip_address(ip_file: str) -> str:
    """
    Reads ip address from ip_file. Blocks until the file is created
    
    :returns: IP address
    """
    while True:
        while not os.path.isfile(ip_file):
            logger.info('waiting for ip file to be created')
            time.sleep(1)
        with open(ip_file, 'r', encoding='utf-8') as f_handle:
            ip_addr = f_handle.readline().strip()
            if ip_addr:
                break
    logger.info(f'read ip {ip_addr} from file {ip_file}')
    return ip_addr


def dump_ip_address(ip_file: str) -> str:
    """
    Dumps the ip address of the current host to a file

    :param ip_file: File name where the ip address of the local host will be dumped
    :type ip_file: str
    
    :returns: A string containing the ip address of the local host
    """
    socket = get_socket()
    host_ip = socket.ip_addr
    with open(ip_file, 'w', encoding='utf-8') as f_handle:
        f_handle.write(host_ip)
    logger.info(f'wrote ip {host_ip} to file {ip_file}')
    return host_ip


class _CommData:  # pylint: disable=too-few-public-methods
    '''
    Namespace for storing data about the communication environment
    '''
    comm_initialized: ClassVar[bool] = False
    comm_device: ClassVar[torch.device]
    rank: ClassVar[int]
    world_size: ClassVar[int]
    master_ip: str
    master_port: int
    backend: str


def nfs_ip_init(_rank: int, ip_file: str) -> str:
    """
    Communicate the ip address of the master machine/worker (with rank = 0) to other
    machines/workers through the file system

    :param _rank: Rank of the current machine
    :type _rank: int
    :param ip_file:  Path to the ip file that will be used to communicate the ip address between workers.\
    The master will write its ip address to this file. Other workers will block until\
    this file is created, and then read the ip address from it.
    :type ip_file: str
    
    :returns:  A string with the ip address of the master machine/worker
    """
    if _rank == 0:
        master_ip = get_ip_address(ip_file)#dump_ip_address(ip_file)
    else:
        master_ip = get_ip_address(ip_file)
    return master_ip


def initialize_comms(_rank: int, _world_size: int, master_ip_address: str,
                     backend: str, _comm_device: Optional[torch.device] = None,
                     master_port_number: int = 1235):
    """
    Initialize Pytorch's communication library

    :param _rank: Rank of the current worker
    :type _rank: int
    :param _world_size: Number of workers. The same as the number of graph partitions
    :type _world_size: int
    :param master_ip_address: IP address of the master worker (worker with rank 0)
    :type master_ip_address: str
    :param backend: Backend to use. Can be ccl, nccl, mpi or gloo
    :type backend: str
    :param _comm_device:  The device on which the tensors should be on in order to transmit them\
    through the backend. If not provided, the device is infered based on the backend type
    :type _comm_device: torch.device
    :param master_port_number:  The port number on the master
    :type _comm_device: int


    """
    assert backend in ['ccl', 'nccl', 'mpi', 'gloo'],\
        'backend must be ccl, nccl, mpi or gloo'
    if _comm_device is None:
        if backend == 'nccl':
            _comm_device = torch.device('cuda')
        else:
            _comm_device = torch.device('cpu')

    if backend == 'ccl':
        # pylint: disable=unused-import
        try:
            import oneccl_bindings_for_pytorch   # type: ignore
        except:
            try:
                import torch_ccl  # type: ignore
            except:
                raise ImportError("None of the oneccl_bindings_for_pytorch and torch_ccl package has been found")

    if not dist.is_initialized():
        os.environ['MASTER_ADDR'] = master_ip_address
        os.environ['MASTER_PORT'] = str(master_port_number)

        socket = get_socket()
        os.environ['TP_SOCKET_IFNAME'] = socket.name
        os.environ['GLOO_SOCKET_IFNAME'] = 'ens5'
        os.environ['CCL_SOCKET_IFNAME'] = socket.name
        os.environ['NCCL_SOCKET_IFNAME'] = socket.name

        os.environ['FI_VERBS_IFACE'] = socket.name
        os.environ['FI_mlx_IFACE'] = socket.name

        os.environ['MPI_COMM_WORLD'] = str(_world_size)
        os.environ['MPI_COMM_RANK'] = str(_rank)

        os.environ['OMPI_COMM_WORLD'] = str(_world_size)
        os.environ['OMPI_COMM_RANK'] = str(_rank)

        os.environ['IMPI_COMM_WORLD'] = str(_world_size)
        os.environ['IMPI_COMM_RANK'] = str(_rank)

        os.environ['I_MPI_COMM_WORLD'] = str(_world_size)
        os.environ['I_MPI_COMM_RANK'] = str(_rank)
        print('rank', _rank,'world_size', _world_size)
        try:
            dist.init_process_group(
                backend=backend, rank=_rank, world_size=_world_size)
        except:
            logger.error("SAR was unable to initialize torch.distributed process group. "
                         "You can try to do it manually before calling sar.initialize_comms")
            raise
    else:
        assert dist.get_backend() in ['ccl', 'nccl', 'mpi', 'gloo'],\
            'backend must be ccl, nccl, mpi or gloo'

    _CommData.rank = _rank
    _CommData.world_size = _world_size
    _CommData.comm_device = _comm_device
    _CommData.comm_initialized = True
    _CommData.master_ip = master_ip_address
    _CommData.master_port = master_port_number
    _CommData.backend = backend

    logger.info('dist initialized')


def is_initialized() -> bool:
    '''
    True if communication has been initialized
    '''
    return _CommData.comm_initialized


def rank() -> int:
    '''
    Get rank of current host
    '''
    assert is_initialized()
    return _CommData.rank


def world_size() -> int:
    '''
    Get world size of the current distributed setup
    '''
    assert is_initialized()
    return _CommData.world_size


def master_port() -> int:
    '''
    Get the master port of the current distributed setup
    '''
    assert is_initialized()
    return _CommData.master_port


def master_ip() -> str:
    '''
    Get the master ip address of the current distributed setup
    '''
    assert is_initialized()
    return _CommData.master_ip


def backend() -> str:
    '''
    Get the backend of the current distributed setup
    '''
    assert is_initialized()
    return _CommData.backend


def comm_device() -> torch.device:
    '''
    Gets the preferred device for the current communication
    backend. For example cpu device for gloo or OneCCL, or
    cuda device for NCCL
    '''
    assert is_initialized()
    return _CommData.comm_device


def all_to_all(recv_tensors: List[torch.tensor], send_tensors: List[torch.tensor],
               move_to_comm_device: bool = False) -> None:
    '''
    wrapper around dist.all_to_all
    '''
    recv_tensors = [x.new(1, *x.size()[1:]) if x.numel()
                    == 0 else x for x in recv_tensors]
    send_tensors = [x.new(1, *x.size()[1:]) if x.numel()
                    == 0 else x for x in send_tensors]

    if move_to_comm_device:
        recv_tensors_cd = [recv_tensor.to(comm_device())
                           for recv_tensor in recv_tensors]
        send_tensors_cd = [send_tensor.to(comm_device())
                           for send_tensor in send_tensors]
        all_to_all_rounds(recv_tensors_cd, send_tensors_cd)
        for recv_tensor, recv_tensor_cd in zip(recv_tensors, recv_tensors_cd):
            recv_tensor.copy_(recv_tensor_cd)
    else:
        all_to_all_rounds(recv_tensors, send_tensors)


def all_reduce(red_tensor: torch.tensor, op: dist.ReduceOp,
               move_to_comm_device: bool = False):   # pylint: disable=invalid-name
    """
    Wrapper around dist.all_reduce

    :param red_tensor: reduction tensor
    :type red_tensor: torch.tensor
    :param op: reduce operation
    :type op: dist.ReduceOp
    :param move_to_comm_device: Move to comm device or not
    :type move_to_comm_device: bool
    """

    if move_to_comm_device:
        red_tensor_cd = red_tensor.to(comm_device())
        dist.all_reduce(red_tensor_cd, op)
        red_tensor.copy_(red_tensor_cd)
    else:
        dist.all_reduce(red_tensor, op)


def all_to_all_rounds(recv_tensors: List[torch.tensor], send_tensors: List[torch.tensor]):
    """
    All_to_all wrapper which breaks down the collective call into multiple
    torch.distributed.all_to_all calls so that the size of the data in each
    call is below Config.max_collective_size
    
    :param recv_tensors: List of tensors to receive from other workers
    :type recv_tensors: List[torch.tensor]
    :param send_tensors: List of tensor to send to other workers
    :type send_tensors: List[torch.tensor]
    """
    if Config.max_collective_size == 0:
        all_to_all_gloo_support(recv_tensors, send_tensors)
    else:
        max_n_elems = Config.max_collective_size
        total_elems = sum(r_tensor.numel() for r_tensor in recv_tensors) + \
            sum(s_tensor.numel() for s_tensor in send_tensors)
        n_rounds_t = torch.tensor(max(1, total_elems // max_n_elems))
        all_reduce(n_rounds_t, dist.ReduceOp.MAX, move_to_comm_device=True)
        n_rounds = int(n_rounds_t.item())
        logger.debug(f'all to all using {n_rounds}')
        for round_idx in range(n_rounds):
            send_tensors_slices = [_get_tensor_slice(s_tensor, n_rounds, round_idx) for
                                   s_tensor in send_tensors]
            recv_tensors_slices = [_get_tensor_slice(r_tensor, n_rounds, round_idx) for
                                   r_tensor in recv_tensors]
            all_to_all_gloo_support(recv_tensors_slices, send_tensors_slices)


def all_to_all_gloo_support(recv_tensors: List[torch.tensor], send_tensors: List[torch.tensor]):
    """
    Since gloo backend doesn't support all_to_all function, SAR implements it
    with multiple asynchronous sends (torch.dist.isend). For every other backend
    torch.dist.all_to_all is used.

    :param recv_tensors: List of tensors to receive from other workers
    :type recv_tensors: List[torch.tensor]
    :param send_tensors: List of tensor to send to other workers
    :type send_tensors: List[torch.tensor]
    """
    if backend() == 'gloo':
        send_requests = []
        for i in range(world_size()):
            if i == rank():
                recv_tensors[i].copy_(send_tensors[i])
            else:
                send_request = dist.isend(send_tensors[i], i)
                send_requests.append(send_request)
        for i in range(world_size()):
            if i != rank():
                dist.recv(recv_tensors[i], i)
        dist.barrier()
    else:
        dist.all_to_all(recv_tensors, send_tensors)


def _get_tensor_slice(tens: tensor, n_splits: int, split_idx: int) -> tensor:
    chunk_size = max(1, tens.size(0) // n_splits)
    start_idx = chunk_size * split_idx
    if split_idx == n_splits-1:
        end_idx = tens.size(0)
    else:
        end_idx = chunk_size * (split_idx + 1)
    start_idx = min(start_idx, tens.size(0) - 1)
    end_idx = min(end_idx, tens.size(0))
    return tens[start_idx: end_idx]


def exchange_single_tensor(recv_idx: int, send_idx: int,
                           recv_tensor: tensor, send_tensor: tensor) -> None:
    """    Sends send_tensor to worker send_idx and fills recv_tensor with data received
    from worker recv_idx. 

    :param recv_idx: index of the worker from which to receive data
    :type recv_idx: int
    :param send_idx: index of the worker to send send_tensor to
    :type send_idx: int
    :param recv_tensor: tensor to receive data from worker recv_idx. Ensure that this tensor \
    has the same shape as the tensor sent by the remote worker
    :type recv_tensor: tensor
    :param send_tensor: tensor to send to the remote worker
    :type send_tensor: tensor
    """
    logger.debug(
        f'{rank()} : exchange_single_tensor on device {send_tensor.device} : {recv_idx}, {send_idx},{recv_tensor.size()},{send_tensor.size()}')
    dtype = send_tensor.dtype
    if send_idx == recv_idx == rank():
        recv_tensor.copy_(send_tensor)
    elif backend() == 'gloo':
        send_request = dist.isend(send_tensor.to(comm_device()), send_idx)
        dist.recv(recv_tensor.to(comm_device()), recv_idx)
        dist.barrier()
    else:
        send_tensors_list = [torch.tensor([1.0], dtype=dtype).to(comm_device())
                             for _ in range(world_size())]

        recv_tensors_list = [torch.tensor([1.0], dtype=dtype).to(comm_device())
                             for _ in range(world_size())]

        active_recv_tensor = recv_tensor.to(comm_device())
        active_send_tensor = send_tensor.to(comm_device())

        recv_tensors_list[recv_idx] = active_recv_tensor
        send_tensors_list[send_idx] = active_send_tensor

        all_to_all(recv_tensors_list, send_tensors_list)

        if active_recv_tensor is not recv_tensor and recv_tensor.size(0) > 0:
            recv_tensor.copy_(active_recv_tensor)

    logger.debug(
        f'{rank()} : done exchange_single_tensor : {recv_idx}, {send_idx},{recv_tensor.size()},{send_tensor.size()}')


def exchange_tensors(tensors: List[torch.tensor], recv_sizes: Optional[List[int]] = None) -> List[torch.tensor]:
    """    tensors is a list of size WORLD_SIZE. tensors[i] is sent to worker i.
    Returns a list of tensors recv_tensors, where recv_tensors[i] is the tensor
    received from worker i. Optionally, you can provide recv_sizes to specify the 
    sizes of the tensors to be received. If recv_sizes is not provided then an initial
    communication round is used to exchange the the tensor sizes before sending the actual
    tensors.


    :param tensors: tensors to send. tensors[i] is sent to worker i
    :type tensors: List[torch.tensor]
    :param recv_sizes: The sizes of the tensors to be received. recv_sizes[i]\
    is the size of the tensor that will be received from worker i.
    :type recv_sizes: Optional[List[int]]
    :returns: A list of received tensors. The ith tensors is the tensor that was\
    received from worker i.

    """
    trailing_dimensions = tensors[0].size()[1:]
    dtype = tensors[0].dtype
    assert all(x.size()[
               1:] == trailing_dimensions for x in tensors[1:]), 'mismatched size tensors'
    assert all(
        x.dtype == dtype for x in tensors[1:]), 'mismatched type tensors'

    tensors_comm_device = [x.to(comm_device()) for x in tensors]

    if recv_sizes is None:
        all_my_sizes = [torch.tensor([x.size(0)], dtype=torch.long).to(
            comm_device()) for x in tensors]
        all_their_sizes = [torch.tensor([-1], dtype=torch.long).to(
            comm_device()) for _ in range(len(tensors))]
        all_to_all(all_their_sizes, all_my_sizes)

        all_their_sizes_i = [cast(int, x.item()) for x in all_their_sizes]
    else:
        all_their_sizes_i = recv_sizes

    all_their_sizes_aug = [max(1, x) for x in all_their_sizes_i]
    recv_tensors = [torch.empty(x, *trailing_dimensions,
                                dtype=dtype).to(comm_device()).fill_(-1) for x in all_their_sizes_aug]
    all_to_all(recv_tensors, tensors_comm_device)

    return [x[:s].to(tensors[0].device) for s, x in zip(all_their_sizes_i, recv_tensors)]


def sync_params(model: torch.nn.Module):
    """Synchronize the model parameters across all workers. The model parameters
    of worker 0 (the master worker) are copied to all workers

    :param model: The model whose parameters are to be synchronized.\
    The model architecture should be the same in all workers.
    :type model: torch.nn.Module

    """
    state_dict = model.state_dict()
    for _, s_v in state_dict.items():
        if rank() != 0:
            s_v.data.zero_()
        all_reduce(s_v.data, op=dist.ReduceOp.SUM, move_to_comm_device=True)


def gather_grads(model: torch.nn.Module):
    """Sum the parameter gradients from all workers. This should be called
    before optimizer.step

    :param model: The model whose parameter gradients are to be synchronized (summed) across all workers.\
    The model architecture should be the same in all workers.
    :type model: torch.nn.Module

    """

    for param in model.parameters():
        if param.grad is not None:
            all_reduce(param.grad, op=dist.ReduceOp.SUM,
                       move_to_comm_device=True)


class CommThread:
    '''
    A general worker thread
    '''

    def __init__(self) -> None:
        self.task_queue: queue.Queue = queue.Queue()
        self.result_queue: queue.Queue = queue.Queue()

        _comm_thread = threading.Thread(target=self._fetch_tasks)
        _comm_thread.daemon = True
        _comm_thread.start()

    def submit_task(self, task_id: str, task: Callable[[], Any]) -> None:
        '''
        Submit a task in the form of a  callable with no arguments.
        '''
        logger.debug('task submitted %s', task_id)
        self.task_queue.put((task_id, task))

    def get_result(self, block: bool = True) -> Any:
        '''
        Reads the result queue and returns the result of the oldest
        executed task whose reult has not been read yet
        '''
        t_1 = time.time()
        res = self.result_queue.get(block=block)
        logger.debug('task result retreival done in %s ', time.time() - t_1)
        return res

    def _fetch_tasks(self) -> None:
        while True:
            _, task = self.task_queue.get()
            result = task()
            if result is not None:
                self.result_queue.put(result)


comm_thread = CommThread()
