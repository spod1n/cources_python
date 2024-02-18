from kubernetes import client, config

config.load_kube_config()

v1 = client.CoreV1Api

container = client.V1Container(
    name='mypythoncontainer',
    image='nginx:latest',
    ports=[client.V1ContainerPort(container_port=80)]
)

pod_spec = client.V1PodSpec(containers=[container])
pod_template = client.V1PodTemplateSpec(spec=pod_spec)

metadata = client.V1ObjectMeta(name='mypod')

pod = client.V1Pod(
    metadata=metadata,
    spec=pod_template
)

v1.create_namespaced_pod(body=pod, namespace='default')

print('pod успішно створено')