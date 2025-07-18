---
# You can also start simply with 'default'
theme: seriph
# random image from a curated Unsplash collection by Anthony
# like them? see https://unsplash.com/collections/94734566/slidev
background: https://cover.sli.dev
# apply unocss classes to the current slide
class: text-center
# https://sli.dev/features/drawing
drawings:
  persist: false
# slide transition: https://sli.dev/guide/animations.html#slide-transitions
transition: slide-left
# enable MDC Syntax: https://sli.dev/features/mdc
mdc: true
# open graph
# seoMeta:
#  ogImage: https://cover.sli.dev
---

# Bayesian Buccaneers

Arjuna James, Tyrone Nicholas, Ben Williams


---
transition: slide-left
---

# Time Spent This Week

```mermaid
pie 
  "Coding models": 50
  "Waiting for training": 10
  "OOMs": 10
  "Building C++ code": 10
  "Debugging TRL": 20
```

---
transition: slide-left
---

# Money Spent This Week


<img src="./images/RunPod billing.png" alt="RunPod expenses" width="540" />

---
transition: slide-left
---

# Things I Learned This Week

<ul>
<v-click><li>There's a another room at the bottom of the fire escape</li></v-click>
<v-click><li>TRL kind of sucks</li></v-click>
<v-click><li>Prompts matter</li></v-click>
<v-click><li>Warmup and learning rate are decisive</li></v-click>
<v-click><li>Use less data</li></v-click>
</ul>


---
transition: slide-left
---

# Demo

https://ujt5b5x9f0hsw6-7860.proxy.runpod.net/

---
transition: slide-left
---

# ChatGPT + PPO: How It Started...

<img src="./images/MangaCreepy.png" alt="Creepy Manga" width="540" />

---
transition: slide-left
---

# After 2 Hours of Debugging...

<img src="./images/SeriousGakuto.png" alt="Serious Face" width="540" />

---
transition: slide-left
---

# After 4 Hours of Debugging...

<img src="./images/Gun.gif" alt="Suicide GIF" width="540" />

---
transition: slide-left
---

# After 6 Hours of Debugging...

<img src="./images/Angry.gif" alt="Angry GIF" width="540" />

---
transition: slide-left
---

# How It Ended

<img src="./images/SFT-PPO-Model-Comp.png" alt="PPO" width="1024" />

---
transition: slide-left
---

<img src="./images/Pain.jpeg" alt="Pain" width="540" />

---
transition: slide-left
---

<h1 style="font-size:2rem;">
  <a href="http://65.109.84.92:7000/" target="_blank">🇨🇳 Str8 Outta China</a>
</h1>
<a href="http://65.109.84.92:7000/" target="_blank">
  <img src="./images/AJ_UI.png" alt="UI Screenshot" width="540" />
</a>

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

<div class="grid grid-cols-2 gap-4 items-start">

  <div class="space-y-2">
    <ul>
      <li>No.10</li>
      <li>MLX plug</li>
    </ul>
    <img src="./images/ben_no10.png" alt="No10 photo" width="200" />
  </div>

  <div>
    <video src="./images/ben_vid.mp4" autoplay loop muted playsinline class="w-full h-auto"></video>
  </div>

</div>