import asyncio
from threading import Thread

import click

from _tpuswarm import manage_tpus, sslcontext, options, run_flask_app


@click.command()
@click.option('--region', prompt="GCP region", default="us-central2-b")
@click.option('--project', prompt='GCP project', default="your-gcp-project")
@click.option('--tpu-device', prompt='TPU device type', default="v4-8")
@click.option('--node-count', prompt='TPU node count', default=4)
@click.option('--batch', prompt='Batch size', default=512)
@click.option('--command', prompt='Command to run on TPU', default='echo "Hello, TPU!" > /tmp/hello.txt')
@click.option('--port', prompt='Port', default=5000)
@click.option('--host', prompt='Host', default='0.0.0.0')
def run(region, project, tpu_device, node_count, batch, command, port, host):
    click.echo(f"Creating TPU swarm in {region} region, project {project}, with {node_count} {tpu_device} devices.")
    options.update({'batch_size': batch, 'project': project, 'region': region, 'tpu_device': tpu_device, 'node_count': node_count, 'command': command})

    loop = asyncio.get_event_loop()
    loop.run_in_executor(None, run_flask_app, host, port)
    loop.run_until_complete(manage_tpus(project, region, tpu_device, node_count, command))


if __name__ == '__main__':
    run()
