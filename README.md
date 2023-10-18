Code for **Emergence of collective open-ended exploration from
Decentralized Meta-Reinforcement learning**

We train two decentralized agents together on an open ended tasks space to study the emergence of collective exploration behaviors. Our agents are able to generalize to novel objects and tasks, as well as an essentially open ended setting. For videos of the trained agents acting in the environment see https://sites.google.com/view/collective-open-ended-explore/

**Training**
To launch the training from scratch on CPU with 16 environments on each of the 8 workers with a total batch size of 128000 environment steps, run:
```
python3 mp_train.py --num_workers 8 --num_envs 16 --rollout_steps 128000
```


**Evaluation**
To launch the evaluation of pretrained models with 8 environments on 1 worker with a total batch size of 8000 enivornemnt steps and record a video, run:
```
python3 --pretrained models --num_workers 1 --num_envs 8 --rollout_steps 8000 --record_video True
```

