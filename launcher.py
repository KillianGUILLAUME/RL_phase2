import boto3
import time
import os
import subprocess
import threading
from fabric import Connection
from dotenv import load_dotenv

load_dotenv()

# --- CONFIG ---
AWS_REGION = os.getenv("AWS_REGION")
AMI_ID = os.getenv("AMI_ID")
INSTANCE_TYPE = os.getenv("INSTANCE_TYPE")
KEY_NAME = os.getenv("KEY_NAME")
KEY_PATH = os.getenv("PEM_KEY_PATH")

LOCAL_DOWNLOAD_PATH = os.getenv("LOCAL_DOWNLOAD_PATH") or os.path.join(os.getcwd(), "RECUP_AUTO")
LOCAL_PROJECT_PATH = os.getcwd()
REMOTE_DIR = "/home/ubuntu/poker_bot"

def background_sync_task(ip_address, key_path, remote_dir, local_path, stop_event):
    """ 
    Sauvegarde silencieuse et intelligente.
    --ignore-existing : Si le fichier est déjà là, on ne le retélécharge PAS.
    """
    print(f"🔄 [AUTO-SYNC] Activé (Toutes les 5min)...")
    os.makedirs(local_path, exist_ok=True)
    os.makedirs(os.path.join(local_path, "models"), exist_ok=True)
    os.makedirs(os.path.join(local_path, "logs"), exist_ok=True)

    while not stop_event.is_set():
        try:
            # 1. Modèles : On ignore ceux qu'on a déjà (--ignore-existing)
            subprocess.run([
                "rsync", "-az", "--ignore-existing", 
                "-e", f"ssh -i {key_path} -o StrictHostKeyChecking=no -o ConnectTimeout=10",
                f"ubuntu@{ip_address}:{remote_dir}/models/",
                os.path.join(local_path, "models")
            ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

            # 2. Logs : Idem
            subprocess.run([
                "rsync", "-az", "--ignore-existing",
                "-e", f"ssh -i {key_path} -o StrictHostKeyChecking=no -o ConnectTimeout=10",
                f"ubuntu@{ip_address}:{remote_dir}/logs/",
                os.path.join(local_path, "logs")
            ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except:
            pass
        
        # Pause de 5 minutes (300s)
        if stop_event.wait(300):
            break

def main():
    if not os.path.exists("requirements.txt"):
        print(f"❌ ERREUR : Pas de requirements.txt ici.")
        return

    ec2_client = boto3.client('ec2', region_name=AWS_REGION)
    ec2 = boto3.resource('ec2', region_name=AWS_REGION)
    instance = None
    ip_address = None
    stop_sync_event = threading.Event()

    try:
        # --- 1. SETUP ---
        try:
            sg = ec2_client.describe_security_groups(GroupNames=["Poker-SSH-Access"])['SecurityGroups'][0]
            sg_id = sg['GroupId']
        except:
            vpc_id = ec2_client.describe_vpcs()['Vpcs'][0]['VpcId']
            sg = ec2_client.create_security_group(GroupName="Poker-SSH-Access", Description="SSH", VpcId=vpc_id)
            sg_id = sg['GroupId']
            ec2_client.authorize_security_group_ingress(GroupId=sg_id, IpPermissions=[{'IpProtocol': 'tcp', 'FromPort': 22, 'ToPort': 22, 'IpRanges': [{'CidrIp': '0.0.0.0/0'}]}])
        
        print(f"🚀 Lancement sur {INSTANCE_TYPE}...")

        # --- 2. LANCEMENT INSTANCE ---
        instances = ec2.create_instances(
            ImageId=AMI_ID, InstanceType=INSTANCE_TYPE, KeyName=KEY_NAME, MinCount=1, MaxCount=1,
            BlockDeviceMappings=[{'DeviceName': '/dev/sda1', 'Ebs': {'VolumeSize': 50, 'VolumeType': 'gp3', 'DeleteOnTermination': True}}],
            # InstanceMarketOptions={'MarketType': 'spot', 'SpotOptions': {'SpotInstanceType': 'one-time'}}, 
            SecurityGroupIds=[sg_id],
            TagSpecifications=[{'ResourceType': 'instance', 'Tags': [{'Key': 'Name', 'Value': 'Poker-Bot-Prod'}]}]
        )
        instance = instances[0]
        instance.wait_until_running()
        instance.reload()
        ip_address = instance.public_ip_address
        print(f"✅ IP : {ip_address}")

        print("🔌 Attente SSH (30s)...")
        time.sleep(30)
        conn = Connection(host=ip_address, user="ubuntu", connect_kwargs={"key_filename": KEY_PATH})

        # --- 3. ENVOI FICHIERS ---
        print("📂 Envoi du code...")
        subprocess.run([
            "rsync", "-az",
            "-e", f"ssh -i {KEY_PATH} -o StrictHostKeyChecking=no",
            "--exclude", "venv", "--exclude", "__pycache__", "--exclude", ".env", "--exclude", "RECUP_AUTO",
            os.path.join(LOCAL_PROJECT_PATH, ""), 
            f"ubuntu@{ip_address}:{REMOTE_DIR}"
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)

        # --- 4. INSTALLATION ---
        print("🛠️ Installation silencieuse...")
        conn.run("sudo apt update -qq && sudo apt install python3-pip python3-venv -y -qq", hide=True)
        conn.run(f"cd {REMOTE_DIR} && python3 -m venv venv")
        conn.run(f"cd {REMOTE_DIR} && source venv/bin/activate && pip install -q -r requirements.txt && pip install -q tqdm rich")

        # --- 5. LANCEMENT SYNC ---
        sync_thread = threading.Thread(
            target=background_sync_task,
            args=(ip_address, KEY_PATH, REMOTE_DIR, LOCAL_DOWNLOAD_PATH, stop_sync_event),
            daemon=True 
        )
        sync_thread.start()

        # --- 6. ENTRAÎNEMENT ---
        print("🔥 Lancement Entraînement...")
        # log_interval=1000 doit être réglé dans trainer_sb3.py pour réduire l'affichage des stats
        conn.run(f"cd {REMOTE_DIR} && source venv/bin/activate && python -m training.trainer_sb3", pty=True)

    except KeyboardInterrupt:
        print("\n🛑 ARRÊT MANUEL (Ctrl+C)")
        
    except Exception as e:
        print(f"❌ ERREUR : {e}")

    finally:
        print("\n🧹 Nettoyage...")
        if stop_sync_event: stop_sync_event.set()
        
        if ip_address:
            try:
                # Dernière récup (avec ignore-existing aussi)
                subprocess.run(["rsync", "-az", "--ignore-existing", "-e", f"ssh -i {KEY_PATH} -o StrictHostKeyChecking=no", f"ubuntu@{ip_address}:{REMOTE_DIR}/models/", os.path.join(LOCAL_DOWNLOAD_PATH, "models")], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=30)
            except: pass

        if instance:
            print("💀 Destruction instance...")
            instance.terminate()
            print("✅ Terminée.")

if __name__ == "__main__":
    main()