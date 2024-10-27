import asyncio

import click

from _tpuswarm import list_queued_tpus, remove_tpu


async def clean_tpu(region, project):
    queues = (await list_queued_tpus(project, region))[0].queued_resources
    tpuswarm_queues = [q for q in queues if q.name.startswith(f"projects/{project}/locations/{region}/queuedResources/tpuswarm-queue-")]
    for node in tpuswarm_queues:
        print(f"Removing node {node.name}")
        await remove_tpu(node)

@click.command()
@click.option('--region', prompt="GCP region", default="us-central2-b")
@click.option('--project', prompt='GCP project', default="your-project")
def run(region, project):
    loop = asyncio.get_event_loop()
    loop.run_until_complete(clean_tpu(region, project))

if __name__ == '__main__':
    run()
