"""
Configures AWS EC2 machine for the experiments. The script does the following:
- Instantiates g4dn.xlarge EC2 with Ubuntu 20.04 and 64 GB storage
- Configures aws_cli
- Installs 11.6 CUDA driver
- Installs Miniconda
- Clones the fosi project
- Installs with pip the fosi/experiments/experiments_requirements.txt file and CUDA toolkit

Requirements:
(1) Python >= 3.9
(2) AWS account; after opening the account, the user must create AWS IAM user with
    AdministratorAccess permissions. The full steps to create the IAM user:
        1. In https://console.aws.amazon.com/iam/ select "Users" on the left and press "Add users".
        2. Provide user name, e.g. fosi, and mark the "Access key" checkbox and press "Next: Permissions".
        3. Expand the "Set permissions boundary" section and select "Use a permissions boundary to control the maximum user permissions".
           Use the Filter to search for AdministratorAccess and then select AdministratorAccess from the list and press "Next: Tags".
        4. No need to add tags so press "Next: Review".
        5. In the next page press "Create user".
        6. Press "Download.csv" to download the new_user_credentials.csv file.
        7. Place the new_user_credentials.csv file in the same folder with this script.
(3) Install the Python pakages boto3, paramiko, botocore, and pandas.
After completing these steps, re-run this script.
After the script finishes the user can connect (via ssh) to the EC2 machine and run the experiments, for example:
python ~/fosi/examples/fosi_jax_example.py
"""

import time
import boto3
import pandas as pd
import paramiko
from botocore.client import ClientError
import os
import json


def get_default_security_group(region, session):
    ec2_client = session.client('ec2', region_name=region)
    response = ec2_client.describe_security_groups()
    for sg in response['SecurityGroups']:
        if sg['Description'] == 'default VPC security group':
            sg_id = sg['GroupId']
    # _ = ec2_client.describe_vpcs()
    response = ec2_client.describe_subnets()
    subnet_id = response['Subnets'][0]['SubnetId']
    print('Found sg and subnet for region', region, ': sg_id:', sg_id, 'subnet_id:', subnet_id)
    return sg_id, subnet_id


def create_ingress_rule(region, session, security_group_id):
    """
    Creates a security group ingress rule with the specified configuration.
    """
    vpc_client = session.client("ec2", region_name=region)
    try:
        response = vpc_client.authorize_security_group_ingress(
            GroupId=security_group_id,
            IpPermissions=[{
                'IpProtocol': 'tcp',
                'FromPort': 10,
                'ToPort': 65535,
                'IpRanges': [{
                    'CidrIp': '0.0.0.0/0'
                }]
            }])
        print("Added inbound rule to sg", security_group_id, "response:\n", response)

    except ClientError as e:
        if e.response["Error"]["Code"] == "InvalidPermission.Duplicate":
            print("Inbound rule for TCP port-range 10-65535 already exists.")
        else:
            print('Could not create ingress security group rule.')
            raise


def create_iam_role(region, role_name, session):
    iam_client = session.client('iam', region_name=region)

    role_policy = {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Sid": "",
                "Effect": "Allow",
                "Principal": {
                    "Service": "lambda.amazonaws.com"
                },
                "Action": "sts:AssumeRole"
            },
            {
                "Sid": "",
                "Effect": "Allow",
                "Principal": {
                    "Service": "ec2.amazonaws.com"
                },
                "Action": "sts:AssumeRole"
            },
            {
                "Sid": "",
                "Effect": "Allow",
                "Principal": {
                    "Service": "events.amazonaws.com"
                },
                "Action": "sts:AssumeRole"
            }
        ]
    }
    try:
        response = iam_client.create_role(RoleName=role_name, AssumeRolePolicyDocument=json.dumps(role_policy))
        print(response)
        response = iam_client.attach_role_policy(RoleName=role_name, PolicyArn='arn:aws:iam::aws:policy/AdministratorAccess')
        print(response)
        time.sleep(10)
    except iam_client.exceptions.EntityAlreadyExistsException as e:
        print(e)
    role = iam_client.get_role(RoleName=role_name)
    return role


def read_credentials_file():
    credentials_file = os.path.abspath(os.path.dirname(__file__)) + "/new_user_credentials.csv"
    df = pd.read_csv(credentials_file)
    username = df['User name'][0]
    access_key_id = df['Access key ID'][0]
    secret_access_key = df['Secret access key'][0]
    return username, access_key_id, secret_access_key


def create_keypair(ec2_client, keypair_name):
    # Search for existing *.pem file (private key) or create one if not found.
    keypair_file_name = os.path.abspath(os.path.dirname(__file__)) + "/" + keypair_name + '.pem'
    try:
        # Create new key pair. Override local pem file if exists.
        response = ec2_client.create_key_pair(KeyName=keypair_name)
        with open(keypair_file_name, 'w') as private_key_file:
            private_key_file.write(response['KeyMaterial'])
            private_key_file.close()
        print("Created key-pair", keypair_name)
    except ClientError as e:
        if e.response["Error"]["Code"] == "InvalidKeyPair.Duplicate":
            print("key-pair", keypair_name, "already exists")
            # If there is also a local pem file, just continue. Otherwise, delete the key-pair, create a new one, and write it to a local pem file.
            if not os.path.exists(keypair_file_name):
                ec2_client.delete_key_pair(KeyName=keypair_name)
                print("Deleted key-pair", keypair_name)
                response = ec2_client.create_key_pair(KeyName=keypair_name)
                with open(keypair_file_name, 'w') as private_key_file:
                    private_key_file.write(response['KeyMaterial'])
                    private_key_file.close()
                print("Created key-pair", keypair_name)
        else:
            raise
    # TODO 2: in case there is no local pem file but there is key-pair with this name - delete it and create new one
    key = paramiko.RSAKey.from_private_key_file(keypair_file_name)
    ssh_client = paramiko.SSHClient()
    ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    return key, ssh_client


def create_instance(ec2_client, keypair_name):
    # Note: this AMI is only available in us-west-2 region!!! If changing the default region need to change the AMI accordingly.
    instances = ec2_client.run_instances(
        BlockDeviceMappings=[
            {
                'DeviceName': '/dev/sda1',
                'Ebs': {
                    'VolumeSize': 64,
                    'VolumeType': 'gp2'
                },
            },
        ],
        ImageId="ami-00712dae9a53f8c15",  # Ubuntu Server 20.04 LTS (HVM),EBS General Purpose (SSD) Volume Type - ami-00712dae9a53f8c15 (64-bit (x86))
        MinCount=1,
        MaxCount=1,
        InstanceType="g4dn.xlarge",
        KeyName=keypair_name
    )
    instance_id = instances["Instances"][0]["InstanceId"]
    print("EC2 instance ID:", instance_id)
    return instance_id


def get_public_ip(ec2_client, instance_id):
    reservations = ec2_client.describe_instances(InstanceIds=[instance_id]).get("Reservations")

    for reservation in reservations:
        for instance in reservation['Instances']:
            instance_public_ip = instance.get("PublicIpAddress")
            print("EC2 instance public IP:", instance_public_ip)
    return instance_public_ip


def execute_ssh_command(ssh_client, cmd, stdout_verification=None, b_stderr_verification=False, get_pty=False):
    stdin, stdout, stderr = ssh_client.exec_command(cmd, get_pty=get_pty)
    print("Executed command:", cmd)
    print("STDOUT:")
    stdout_str = stdout.read()
    print(stdout_str)
    print("STDERR:")
    stderr_str = stderr.read()
    print(stderr_str)

    if stdout_verification is not None:
        if stdout_verification not in stdout_str.decode():
            print("Verification string", stdout_verification, "not in stdout")
            exit(1)
    if b_stderr_verification:
        if stderr_str != b'':
            print("stderr is not empty.")
            exit(1)


def configure_ec2_instance(region='us-west-2'):
    username, access_key_id, secret_access_key = read_credentials_file()
    session = boto3.session.Session(aws_access_key_id=access_key_id, aws_secret_access_key=secret_access_key)

    ec2_client = session.client('ec2', region_name=region)
    keypair_name = "aws_fosi_private_key"

    try:
        sg_id, subnet_id = get_default_security_group(region, session)
        create_ingress_rule(region, session, sg_id)

        # Create the instance and connect/ssh to it
        key, ssh_client = create_keypair(ec2_client, keypair_name)
        instance_id = create_instance(ec2_client, keypair_name)
        time.sleep(40)  # Wait until the instance obtains its public IP
        instance_public_ip = get_public_ip(ec2_client, instance_id)
        ssh_client.connect(hostname=instance_public_ip, username="ubuntu", pkey=key)

        # Install aws cli
        execute_ssh_command(ssh_client, 'sudo apt-get -y install unzip')
        execute_ssh_command(ssh_client, 'curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"')
        execute_ssh_command(ssh_client, 'unzip awscliv2.zip')
        execute_ssh_command(ssh_client, 'sudo ./aws/install')
        execute_ssh_command(ssh_client, 'aws --version', stdout_verification="aws-cli")

        # Configure aws cli
        execute_ssh_command(ssh_client, 'aws configure set aws_access_key_id ' + access_key_id)
        execute_ssh_command(ssh_client, 'aws configure set aws_secret_access_key ' + secret_access_key)
        execute_ssh_command(ssh_client, 'aws configure set region ' + region)
        execute_ssh_command(ssh_client, 'aws configure set output json')
        execute_ssh_command(ssh_client, 'aws configure get region', stdout_verification=region)  # Verify configuration worked

        # Install CUDA drivers
        execute_ssh_command(ssh_client, 'sudo apt-get update -y')
        execute_ssh_command(ssh_client, 'sudo apt-get upgrade -y linux-aws')
        execute_ssh_command(ssh_client, 'sudo reboot')
        time.sleep(40)  # Wait until the instance reboots
        ssh_client.connect(hostname=instance_public_ip, username="ubuntu", pkey=key)
        execute_ssh_command(ssh_client, 'sudo apt-get install -y gcc make linux-headers-$(uname -r)')
        execute_ssh_command(ssh_client, 'cat << EOF | sudo tee --append /etc/modprobe.d/blacklist.conf\nblacklist vga16fb\nblacklist nouveau\nblacklist rivafb\nblacklist nvidiafb\nblacklist rivatv\nEOF')
        execute_ssh_command(ssh_client, 'echo GRUB_CMDLINE_LINUX="rdblacklist=nouveau" | sudo tee -a /etc/default/grub')
        execute_ssh_command(ssh_client, 'sudo update-grub')
        execute_ssh_command(ssh_client, 'aws s3 cp --recursive s3://ec2-linux-nvidia-drivers/grid-14.0/ .')
        execute_ssh_command(ssh_client, 'chmod +x NVIDIA-Linux-x86_64*.run')
        execute_ssh_command(ssh_client, 'sudo /bin/sh ./NVIDIA-Linux-x86_64*.run -s')
        execute_ssh_command(ssh_client, 'sudo reboot')
        time.sleep(40)  # Wait until the instance reboots
        ssh_client.connect(hostname=instance_public_ip, username="ubuntu", pkey=key)
        execute_ssh_command(ssh_client, 'nvidia-smi -q | head', stdout_verification='11.6')  # Verify configuration worked

        # Install Miniconda
        execute_ssh_command(ssh_client, 'wget https://repo.anaconda.com/miniconda/Miniconda3-py39_4.12.0-Linux-x86_64.sh')
        execute_ssh_command(ssh_client, 'chmod 755 Miniconda3-py39_4.12.0-Linux-x86_64.sh')
        execute_ssh_command(ssh_client, './Miniconda3-py39_4.12.0-Linux-x86_64.sh -b')
        # Update .bashrc file
        execute_ssh_command(ssh_client, "echo '# >>> conda initialize >>>' >> ~/.bashrc")
        execute_ssh_command(ssh_client, "echo '# !! Contents within this block are managed by 'conda init' !!' >> ~/.bashrc")
        execute_ssh_command(ssh_client, "echo '__conda_setup=\"$('/home/ubuntu/miniconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)\"' >> ~/.bashrc")
        execute_ssh_command(ssh_client, "echo 'if [ $? -eq 0 ]; then' >> ~/.bashrc")
        execute_ssh_command(ssh_client, "echo '    eval \"$__conda_setup\"' >> ~/.bashrc")
        execute_ssh_command(ssh_client, "echo 'else' >> ~/.bashrc")
        execute_ssh_command(ssh_client, "echo '    if [ -f \"/home/ubuntu/miniconda3/etc/profile.d/conda.sh\" ]; then' >> ~/.bashrc")
        execute_ssh_command(ssh_client, "echo '        . \"/home/ubuntu/miniconda3/etc/profile.d/conda.sh\"' >> ~/.bashrc")
        execute_ssh_command(ssh_client, "echo '    else' >> ~/.bashrc")
        execute_ssh_command(ssh_client, "echo '        export PATH=\"/home/ubuntu/miniconda3/bin:$PATH\"' >> ~/.bashrc")
        execute_ssh_command(ssh_client, "echo '    fi' >> ~/.bashrc")
        execute_ssh_command(ssh_client, "echo 'fi' >> ~/.bashrc")
        execute_ssh_command(ssh_client, "echo 'unset __conda_setup' >> ~/.bashrc")
        execute_ssh_command(ssh_client, "echo '# <<< conda initialize <<<' >> ~/.bashrc")
        execute_ssh_command(ssh_client, "echo 'export PYTHONPATH=$PYTHONPATH:/home/ubuntu/fosi' >> ~/.bashrc")
        execute_ssh_command(ssh_client, "echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/ubuntu/miniconda3/envs/fosi/lib:/home/ubuntu/miniconda3/pkgs/cudatoolkit-11.3.1-h2bc3f7f_2/lib' >> ~/.bashrc")
        execute_ssh_command(ssh_client, "echo 'export PATH=/home/ubuntu/miniconda3/envs/fosi/bin${PATH:+:${PATH}}' >> ~/.bashrc")
        # Create fosi environment
        execute_ssh_command(ssh_client, 'bash -ic "conda create --name fosi --clone base"')
        execute_ssh_command(ssh_client, "echo 'conda activate fosi' >> ~/.bashrc")

        # Clone the fosi project, pip install experiments/requirements.txt, and pip install CUDA toolkit
        execute_ssh_command(ssh_client, 'git clone https://github.com/hsivan/fosi')
        execute_ssh_command(ssh_client, 'bash -ic "pip install -r fosi/experiments/experiments_requirements.txt -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html"', get_pty=True)
        execute_ssh_command(ssh_client, 'bash -ic "conda install -y -c \"nvidia/label/cuda-11.8.0\" cuda"', get_pty=True)

        ssh_client.close()
    except Exception as e:
        print(e)
        exit(1)
    return instance_public_ip


if __name__ == "__main__":
    configure_ec2_instance()
