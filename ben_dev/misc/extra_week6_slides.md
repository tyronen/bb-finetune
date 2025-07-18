---
transition: slide-left
---

# Week 6 – Hard mode

- Two days lost to induction
- Another half‑day on admin

<img src="./images/ben_drake.png" alt="Drake meme" width="540" />

---
transition: slide-left
---

# JAX

- Decided to spend the week learning JAX for the very first time
- Why the hype?
- Main goal: fine‑tune GPT‑2 with LoRA on my laptop GPU

<img src="./images/ben_jax.png" alt="JAX graph" width="540" />

---
transition: slide-left
---

# What I Achieved in my 2.5 days

- Loaded GPT‑2 with Flax (small enough for my GPU)
- Got it training at a low learning rate
- LoRA integration = package pain
- Finally got LoRA working – loss ⬇️, accuracy ⬆️

<img src="./images/ben_wandb.png" alt="WandB screenshot" width="540" />

---
transition: slide-left
---

# JAX vs PyTorch

- JAX pre‑compiles graphs vs PyTorch rebuilds the graph on every forward pass 
- JAX parallelises across devices automatically (untested)
- JAX is more “functional” – you need wrappers:  
  - **Flax** for NN modules  
  - **Optax** for optimisers  
- No PEFT‑style library, so I used **EasyDeL** (which was not 'Easy')

<img src="./images/ben_easydel.png" alt="EasyDeL pain" width="540" />

---
transition: slide-left
---

# Lessons Learned

- JAX is way more effort than its worth for most use cases 
- Docs aren't good
- "JAX is blazingly fast once it’s working"
- Will I carry on with it? Probably not in the near term

<img src="./images/ben_boxing.png" alt="PyTorch vs JAX" width="540" />

---
transition: slide-left
---

# Humble brag

- No.10
- MLX plug

<img src="./images/ben_no10.png" alt="No10 photo" width="540" />