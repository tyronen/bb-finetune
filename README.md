# Bayesian Buccaneers Week 6

## Setting up your dev environment

Edit ~/.ssh/config to point 'mlx' at your GPU instance, eg

This assumes you have a private key id_ed25519 you use for GitHub and for ssh.

```
Host mlx
  HostName <ip address of your host>
  IdentityFile ~/.ssh/id_ed25519
  IdentitiesOnly yes
  Port <port of your host>
  User root
```

```
# on laptop
./send.sh
ssh mlx
# on GPU
source ssh.sh
```

Have your IDE rsync your Git repo on your laptop to `mlx:/workspace/bb-finetune`. Then you can run `source setup.sh` on
that directory on the GPU. No need to git pull on GPU!

## Launching the script

```
accelerate launch --mixed_precision bf16 week6.py
```