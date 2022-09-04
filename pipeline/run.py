import kfp
from kfp import dsl
from kfp.dsl import ContainerOp
from kfp.components import InputPath
from kubernetes import client as k8s_client
from kfp import compiler

def attach_output_volume(fun):
    """Attaches emptyDir volumes to container operations.
    See https://github.com/kubeflow/pipelines/issues/1654
    """

    def inner(*args, **kwargs):
        op: ContainerOp = fun(*args, **kwargs)
        # op.output_artifact_paths = {
        #     'mlpipeline-ui-metadata': '/output/mlpipeline-ui-metadata.json',
        #     'mlpipeline-metrics': '/output/mlpipeline-metrics.json',
        # }
        op.add_volume(
            k8s_client.V1Volume(name='volume', empty_dir=k8s_client.V1EmptyDirVolumeSource())
        )
        op.container.add_volume_mount(k8s_client.V1VolumeMount(name='volume', mount_path='/output'))

        return op

    return inner


# @attach_output_volume
# def download_data(url: InputPath):
#     return dsl.ContainerOp(
#         name='Download dataset',
#         image='appropriate/curl',
#         command=['sh', '-c'],
#         arguments=['curl', '-Lo', '/output/dataset.tar.gz', url],
#         file_outputs={'data': '/output/dataset.tar.gz'},
#     )

@attach_output_volume
def train(tag):
    return dsl.ContainerOp(
        name='Do trainng',
        image='yunwei37/cloudremoval:%s' % tag,
        file_outputs={'data': '/output/results'},
    )


@dsl.pipeline(
    name='Cloud Removal CPU Training',
    description='A very simple demonstration',
)
def cloud_removal_pipeline(
    tag='0.0.1-with-dataset',
):
    train(tag)

if __name__ == '__main__':
    # Compiling the pipeline
    compiler.Compiler().compile(cloud_removal_pipeline, 'cloudremoval.yaml')