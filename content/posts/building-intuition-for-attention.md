---
title: "Paying attention to ATTENTION"
date : 2025-11-26T12:48:50+05:45
draft : false
tags : ["llm"]
---

I recently stumbled upon a three-part YouTube series that explains the Attention mechanism in a way I hadn’t seen before. The creator walks through the idea with such clarity that it made me pause, rewind, think, and… well, actually understand what’s going on under the hood. After finishing the series, I felt like, “Okay, I need to write this down — partly so I don’t forget it, and partly because explaining something forces me to understand it even more.”

So that’s what this post is. If you know Hindi, I’d genuinely recommend watching the original videos; the creator has put in a level of effort that’s rare on YouTube. What I’m doing here is simply taking the intuition I got from those videos and reinterpreting the same in a way that makes sense to me, and hopefully to you too.

_This post assumes you already have some basic understanding of NLP and ML; otherwise, the explanations would get way too long for our goldfish-level attention spans (mine included — I’m kinda lazy, to write too much lol)._

## Embeddings — The Zeros and Ones of NLP

Before we even get to Attention, we need to talk about embeddings, because they’re the foundation of pretty much everything in modern NLP. If you strip away the hype, embeddings are just a way of turning words into numbers — because, sadly, your GPU doesn’t understand English, Hindi, Nepali, or emoji. It understands numbers, and underneath that, just 0s and 1s.

So the entire NLP journey starts with a simple question:

How do we convert words into meaningful numbers?

## From Counting Words… to Understanding Words

In the early days, our approach was quite simple: count stuff.

How many times does the word appear?
Which words appear near each other?
Can we build a giant sparse vector with counts?

These methods (Bag-of-Words, TF-IDF) were okay for basic tasks, but they treated each word like a unique card — no relationships, no meaning, no nuance.

Then came the first major leap: learn the embeddings instead of designing them.

This gave us Word2Vec, GloVe, and friends — models that learned to place words in a vector space where similar words end up close together. At this point, embeddings stopped being plain numbers and started becoming… well, meaningful.

That’s how we got the now-iconic example:

> vector("queen") – vector("woman") + vector("man") ≈ vector("king")

When you see that for the first time, it genuinely feels like cheating. Suddenly, algebra is capturing actual semantic relationships. (Yes, the same linear algebra you swore you’d never need. Welcome back. It’s here to stay.)

But there is still one issue.

> One word = one vector = one meaning.

But language doesn’t behave like that. Again consider the example:

> “He sat on the river bank.”

> “He deposited cash at the bank.”

Very same set of words, yet completely different meaning.

Old-school embeddings couldn’t handle this. The word “Bank” had a single vector, no matter where it appeared. 

## Intuition Behind Self Attention (Context-Aware Embeddings)

To truly understand language, word representations must change depending on the sentence they appear in.

The word “bank” means very different things in “He sat on the river bank.” versus “He deposited cash at the bank.” Even though the word is the same, its meaning is determined by the surrounding context. This means we need a way to convert “bank” into different vectors depending on the words around it.

This is where self-attention comes into play. Self-attention allows a model to dynamically adjust a word’s embedding by looking at other words in the sentence and weighing their relevance. In effect, the representation of “bank” is reshaped by its context.

The following animation is a simplified view of this process: starting from the base embedding of “bank,” the model shifts that vector as contextual words are introduced, producing a context-aware representation.

<div style="text-align: center;">
    <img src="/img/attention/attention.gif" alt="attention visualization">
</div>

As you can see in the simplified illustration above, the embedding vector for “bank” changes as different surrounding words are introduced. This is the core idea behind attention. Now that we have an intuitive understanding of what attention does, let’s dive into how it that vector transformation actually works.







