---
title: 'Understanding Positional Encodings'
date: 2026-01-24T17:19:38-05:00
draft: true
---

If the following image doesn't make sense to you, there's a good chance you haven't fully grasped positional encodings in transformer architecture. Please stick around as I'll try to explain them in an intuitive way.

<figure style="display: flex; flex-direction: column; justify-content: center; align-items: center;">
    <img src="/img/positional-encoding/crux.png" alt="crux of positional encoding">
    <figcaption style="text-align: center; font-style: italic; margin-top: 8px;">The 128-dimensional positional encoding for a sentence</figcaption>
</figure>

## Context 

This blog is a part of my quest to properly understand the transformer architecture. Previously, I tried to explain self-attention in my own way [here](/posts/paying-attention-to-attention/). This is just a continuation trying to explain positional encoding. Also, as I mentioned in my previous blog, I've taken heavy inspiration (and learning) from [this video](https://youtu.be/GeoQBNNqIbM?si=T-dhKf41DCLuGX49), and if you know Hindi, I would genuinely suggest you to rather go over his playlist.

## Prelude

To understand what positional encodings are, lets zoom into this section of the transformer architecture

<figure style="display: flex; flex-direction: column; justify-content: center; align-items: center;">
    <img src="/img/positional-encoding/positional_encoding_layer.png" alt="crux of positional encoding">
</figure>

Now, lets break down the components you see in the zoomed-in figure.

### Inputs  

These are the words that are fed into the model. Before being fed, these words go through what's called tokenization—a process that breaks text into smaller units called tokens. Tokenization is a vast field in itself, so I won't be covering it in this blog. For simplicity, let's assume each word in the English language has a unique ID associated with it. When we pass a sentence to a transformer, we are actually passing an array of these token IDs as input.

<figure style="display: flex; flex-direction: column; justify-content: center; align-items: center;">
    <img src="/img/positional-encoding/tokenization.gif" alt="tokenization flow">
</figure>


### Input Embedding

The token IDs by themselves are just integers, i.e., they don't carry any semantic meaning. The input embedding layer converts each token ID into a dense vector of fixed dimension (say, 512 or 768). Think of it as a lookup table: each token ID maps to a learnable vector that captures the meaning of that word. These vectors are what the transformer actually works with. Think of it like converting a single number into an n-dimensional vector that has semantic meaning.

<figure style="display: flex; flex-direction: column; justify-content: center; align-items: center;">
    <img src="/img/positional-encoding/embedding.gif" alt="embedding flow">
</figure>


## Positional Encoding

### The need

Now that we understand how an input sentence gets converted into dense vectors, let's understand why we need positional encoding in the first place.

So far, we've converted a list of words into a list of dense vectors. But here's the problem: the embedding for a word is always the same regardless of where it appears in the sentence. The word "love" will have the same vector whether it's the first word or the last.

Consider: "I love transformers" vs "Transformers love I" (yeah you grammar nazis, the second one isn't a grammatically valid sentence, but please bear with me to understand the concept). These two sentences have the exact same words, yet their meanings are completely different. The position of words matters! What if we could inject some positional information into each word's embedding so that the same word at different positions gets a slightly different representation?

### The simplest way

Let's think from first principles. What's the simplest way to add position information to a vector? 

What if we just add the word's index to its embedding? So the first word gets +0 added to all dimensions, the second word gets +1, the third gets +2, and so on. This would shift each word's embedding based on its position in the sentence.

<figure style="display: flex; flex-direction: column; justify-content: center; align-items: center;">
    <img src="/img/positional-encoding/simple_position.gif" alt="simple positional encoding">
</figure>

Simple, right? But there are some major issues with this approach:

1. **Unbounded values**: The number of input words can grow quite large (512, 1024, even 4096 tokens). If we keep adding the word's index, the position values can become massive. Imagine adding 4096 to every dimension of a word embedding, the positional information would completely overpower the actual semantic meaning of the word.

2. **Inconsistent scale**: The first word gets +0, but the 1000th word gets +1000. This huge variation in magnitude makes it hard for the model to learn meaningful patterns.

So ok, then what if we normalize the index before adding it to the word embedding? That should get rid of the above issues right?

For example, we could divide the position by the total sequence length (`N`): position 0 becomes 0, position 1 becomes `1/N`, position 2 becomes `2/N`, and so on. This keeps all values between 0 and 1, so yeah, this kinda solves the above two issues.

But wait, this creates a _new problem_. The same word at the same absolute position would get different positional values depending on the sentence length! In a 10-word sentence, position 5 gets 0.5. In a 100-word sentence, position 5 gets 0.05. The model would have a hard time learning that these represent the "same" position.

Also, both approaches add discrete integers to the embedding. Neural networks generally work better with smooth, continuous representations rather than jumpy discrete values.

Another limitation of raw position indices is that they don’t naturally encode distance. The model knows absolute positions, but understanding how far apart two tokens are requires learning subtraction. Something like a periodic function could help here. Since it repeats its pattern after a fixed interval, shifting from position `p` to `p + k` results in a predictable transformation rather than a completely new value. This makes the relationship between positions easier to capture, allowing the model to learn relative distances through simple transformations instead of explicit arithmetic.

So to summarize this section, we need something that ticks all three boxes:
- **Bounded**: Values stay in a fixed range regardless of position
- **Continuous**: Smooth transitions between positions
- **Periodic**: Can encode relative distances naturally

### Trigonometric Functions


Enter sine and cosine, the perfect candidates! They're bounded between -1 and 1, inherently continuous, and periodic.

<figure style="display: flex; flex-direction: column; justify-content: center; align-items: center;">
    <img src="/img/positional-encoding/sine-and-cosine.png" alt="sine and cosine functions">
</figure>







