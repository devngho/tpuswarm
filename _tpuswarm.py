import os
import random
import ssl

from flask import Flask, request
import asyncio
from google.cloud.tpu_v2alpha1 import TpuAsyncClient, CreateQueuedResourceRequest, QueuedResource, Node, \
    ListNodesRequest, DeleteQueuedResourceRequest, NetworkConfig, QueuedResourceState
import aiohttp
from itertools import cycle, chain

app = Flask(__name__)
client = TpuAsyncClient()
sslcontext = ssl.create_default_context(purpose=ssl.Purpose.CLIENT_AUTH)

available_ips = []
ips = []
setup_ips = []
options = {
    'batch_size': 512,
}
aiohttp_errors = (aiohttp.client_exceptions.ClientResponseError, aiohttp.client_exceptions.ClientConnectorDNSError, aiohttp.client_exceptions.ClientConnectorError)

async def send_request_to_ip(session, ip, data):
    try:
        async with session.post(f"https://{ip}:8080/batch", json=data, timeout=10000) as response:
            return await response.json()
    except Exception as e:
        return e

async def check_node_available(session, ip):
    try:
        async with session.get(f"https://{ip}:8080/heartbeat", timeout=10000) as response:
            return response.status == 200
    except Exception as e:
        print(f"Failed to check node {ip}: {e}")
        return False

@app.post('/batch')
async def batch():
    """
    Input shape: {'prompts': ['prompt1', 'prompt2', ...], 'samplings': {'temperature': 0.7, 'min_p': 0.9, 'max_tokens': 100}}
    Output shape: {'results': ['result1', 'result2', ...]}
    """

    body = request.json
    # split into batches
    batch_size = options['batch_size']
    batches = [body['prompts'][i:i + batch_size] for i in range(0, len(body['prompts']), batch_size)]

    # distribute to available nodes. Bind >1 batches to a node is allowed
    async with aiohttp.ClientSession(trust_env=True, connector=aiohttp.TCPConnector(limit_per_host=5, verify_ssl=False)) as session:
        tasks = [send_request_to_ip(session, ip[1], {
            'prompts': batch,
            'samplings': body['samplings']
        }) for ip, batch in zip(cycle(available_ips), batches)]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # error -> try other nodes
        retry_targets = [i for i, result in enumerate(results) if isinstance(result, Exception) or any(map(lambda x: x in result, aiohttp_errors))]
        if len(retry_targets) == 0: # all success
            # flatten the results
            return {'results': list(chain([result for r in results for result in r['result']]))}

        # retry
        for i in retry_targets:
            for ip in available_ips:
                print(f"Retrying batch {i} on {ip}")
                try:
                    results[i] = await send_request_to_ip(session, ip[1], {
                        'prompts': batches[i],
                        'samplings': body['samplings']
                    })
                    break
                except Exception as e:
                    pass
                except aiohttp_errors as e:
                    pass

    return {'results': list(chain([result for r in results for result in r['result']]))}


async def list_tpus(project, region):
    result = await client.list_nodes(ListNodesRequest(parent=f"projects/{project}/locations/{region}"))
    result_list = []
    async for node in result.pages:
        result_list.append(node)
    return result_list

async def list_queued_tpus(project, region):
    result = await client.list_queued_resources(parent=f"projects/{project}/locations/{region}")
    result_list = []
    async for node in result.pages:
        result_list.append(node)
    return result_list

async def create_tpu(name, project, region, tpu_device):
    node = Node(runtime_version="tpu-ubuntu2204-base", accelerator_type=tpu_device, network_config=NetworkConfig(enable_external_ips=True))

    nodespec = QueuedResource.Tpu.NodeSpec(parent=f"projects/{project}/locations/{region}", node_id=f"tpuswarm-node-{name}", node=node)

    tpu = QueuedResource.Tpu(node_spec=[nodespec])

    rs = QueuedResource(tpu=tpu, spot=QueuedResource.Spot())

    req = CreateQueuedResourceRequest(parent=f"projects/{project}/locations/{region}", queued_resource_id=f"tpuswarm-queue-{name}", queued_resource=rs)

    await client.create_queued_resource(req)

async def launch_command_on_tpu(name, project, region, command):
    # just do it as a shell command
    shell_command = f'gcloud alpha compute tpus tpu-vm ssh {name} --project={project} --zone={region} --command="{command}"'
    os.system(shell_command)

async def remove_tpu(resource):
    await client.delete_queued_resource(DeleteQueuedResourceRequest(name=resource.name, force=True))

async def manage_tpus(project, region, tpu_device, node_count, command):
    print("Starting TPU management loop")

    # work for each 60s
    global setup_ips

    async with aiohttp.ClientSession(trust_env=True, connector=aiohttp.TCPConnector(limit_per_host=5, verify_ssl=False)) as session:
        while True:
            queues = (await list_queued_tpus(project, region))[0].queued_resources
            tpuswarm_nodes = [q for q in queues if q.name.startswith(f"projects/{project}/locations/{region}/queuedResources/tpuswarm-queue-")]

            active_nodes = [q for q in tpuswarm_nodes if q.state.state == QueuedResourceState.State.ACTIVE]
            waiting_nodes = [q for q in tpuswarm_nodes if q.state.state == QueuedResourceState.State.WAITING_FOR_RESOURCES]
            provisioning_nodes = [q for q in tpuswarm_nodes if q.state.state == QueuedResourceState.State.PROVISIONING]
            suspending_nodes = [q for q in tpuswarm_nodes if q.state.state == QueuedResourceState.State.SUSPENDING]
            suspended_nodes = [q for q in tpuswarm_nodes if q.state.state == QueuedResourceState.State.SUSPENDED]

            print(f"TPU nodes: {len(tpuswarm_nodes)}(active: {len(active_nodes)}, waiting: {len(waiting_nodes)}, provisioning: {len(provisioning_nodes)})")

            nodes = (await list_tpus(project, region))[0].nodes
            tpuswarm_nodes_real = [n for n in nodes if n.name.startswith(f"projects/{project}/locations/{region}/nodes/tpuswarm-node-") and n.state == Node.State.READY]
            ips = [(n.name.split('/')[-1], n.network_endpoints[0].access_config.external_ip) for n in tpuswarm_nodes_real]
            # check not setup ips
            setup_required_ips = list(set(ips) - set(setup_ips))
            for name, ip in setup_required_ips:
                print(f"Setup TPU node {name}")

                # if node available, skip
                available = await check_node_available(session, ip)
                if available:
                    print(f"Node {name} already available")
                    setup_ips.append((name, ip))
                    continue

                try:
                    await launch_command_on_tpu(name, project, region, command)
                    setup_ips.append((name, ip))
                except Exception as e:
                    print(f"Failed to setup TPU node: {e}. Continuing.")

            # heartbeat check for each node, then add to available_ips
            tasks = [check_node_available(session, ip) for name, ip in setup_ips]
            results = await asyncio.gather(*tasks)
            res = [ip for ip, result in zip(setup_ips, results) if result]

            available_ips.clear()
            available_ips.extend(res)

            print(f"Available TPU nodes: {len(available_ips)}")

            if len(active_nodes) + len(waiting_nodes) + len(provisioning_nodes) + len(suspending_nodes) + len(suspended_nodes) < node_count: # suspend nodes are included in quota!
                node_id_num = random.randint(1, 100000)
                print(f"Creating TPU node {node_id_num}")
                try:
                    await create_tpu(str(node_id_num), project, region, tpu_device)
                except Exception as e:
                    print(f"Failed to create TPU node: {e}. Continuing.")

            # then kill the suspended nodes
            for node in suspended_nodes:
                node_name = node.tpu.node_spec[0].node_id
                print(f"Removing suspended node {node_name}")
                setup_ips = [ip for ip in setup_ips if ip[0] != node_name]
                await remove_tpu(node)
            await asyncio.sleep(30)
            # await create_tpu("1", project, region, tpu_device)

def run_flask_app(host, port):
    app.run(host=host, port=port, debug=False, use_reloader=False)