---
title: 'Understanding Positional Encodings'
date: 2026-01-24T17:19:38-05:00
draft: true
---

If the following image is not familiar to you or you do not fully understand it, then there is a good chance you might not have fully understood positional encodings in transformer architecture. Please stick around as I'll try to explain positional encodings in an intuitive way.

<figure style="display: flex; flex-direction: column; justify-content: center; align-items: center;">
    <img src="/img/positional-encoding/crux.png" alt="crux of positional encoding">
    <figcaption style="text-align: center; font-style: italic; margin-top: 8px;">The 128-dimensional positional encoding for a sentence</figcaption>
</figure>

## Context 

This blog is a part of my quest to properly understand the transformer architecture. Previously, I tried to explain self-attention in my own way [here](/posts/paying-attention-to-attention/). This is just a continuation trying to explain positional encoding. Also, as I mentioned in my previous blog, I've taken heavy inspiration (and learning) from [this video](https://youtu.be/GeoQBNNqIbM?si=T-dhKf41DCLuGX49), and if you know Hindi, I would genuinely suggest you to rather go over his playlist.

## So what are positional encodings?

To understand what positional encodings are, lets zoom into this section of the transformer architecture

<figure style="display: flex; flex-direction: column; justify-content: center; align-items: center;">
    <img src="/img/positional-encoding/positional_encoding_layer.png" alt="crux of positional encoding">
</figure>

Now, lets break down the components you see in the zoomed-in figure.

### Input    

These are the words that are fed into the model. Before being fed, these words go through what's called tokenization—a process that breaks text into smaller units called tokens. Tokenization is a vast field in itself, so I won't be covering it in this blog. For simplicity, let's assume each word in the English language has a unique ID associated with it. When we pass a sentence to a transformer, we are actually passing an array of these token IDs as input.

