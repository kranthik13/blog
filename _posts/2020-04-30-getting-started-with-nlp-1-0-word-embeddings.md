---
layout: post
title:  "Getting Started with NLP : 1.0 | Word Embeddings |"
date:   2020-05-1 23:14:13 +0530
categories: 
---
## Word Embeddings

Word Embeddings are word representation techniques. They are capable of capturing the context of a word in a document, its semantic and syntactic meaning, similarities with other words, etc.

Loosely speaking, it is a technique which can be used to represent a word in a real-number vector (array of floats) which can capture its meaning and the context. These vectors then can be further used to build Machine Learning models.

Wikipedia's Formal Definition : 
 > **Word embedding** is the collective name for a set of [language modeling](https://en.wikipedia.org/wiki/Language_model "Language model") and [feature learning](https://en.wikipedia.org/wiki/Feature_learning "Feature learning") techniques in [natural language processing](https://en.wikipedia.org/wiki/Natural_language_processing "Natural language processing") (NLP) where words or phrases from the vocabulary are mapped to [vectors](https://en.wikipedia.org/wiki/Vector_(mathematics) "Vector (mathematics)") of [real numbers](https://en.wikipedia.org/wiki/Real_numbers "Real numbers").

![Word Embeddings](https://developers.google.com/machine-learning/crash-course/images/linear-relationships.svg)

### Need
Previously we have seen how we can use the count-based frequency methods to vectorize (loosely : make sense) of textual data so that we can run our Machine Learning Models. 
But most of these methods had recognizable flaws : 
1. The **structure/order of the sentence didn't matter** as they were derived from count-based features and for count it doesn't matter where the word is in the sentence i.e its **position**.
2. Did not consider **semantic nature** i.e what a word **means**, how is it **related to other words**, **similarity**, etc.
3. Didn't consider **context** where the particular word is used. 
	
	- Consider two sentences : 
	
		- A : "deposit money in bank."
		- B : "water flows from one bank to the other."
	
		Here, the 'bank' word needs to have context in order to understand its particular meaning as both the sentences mean differently for the word 'bank'.
		For count-based methods these meant the same as they didn't consider the context in which a particular word is being used.


## Techniques

There are multiple techniques of building word embeddings and using them. Some of the most popular ones are : 

### 1. Word2Vec : 
		
This was the first-of-its-kind word embedding model which used neural networks to build the embedding model. Published by [Google Reasearch Team in 2013](https://arxiv.org/pdf/1310.4546.pdf). This paper consisted of techniques like **Skip-Gram Model** and computational-efficient versions of the same.
Soon **Common Bag of Words (CBOW)** model also got published.

 - **CBOW** : The main principle behind this was **given a context the model had to predict target word**.
		
		"Where are you going Robert?"
		  [0]  [1] [2]  [3]   [4]
	
	Consider a **context window of size 2** which would look like : 

	| Word   | Left-Context  | Right-Context |
	|---|---| --- |
	| you  | [where, are]  | [going, robert]  |
	| going  | [are, you]  | [robert]  |
	| robert  | [you, going]  | [] |

	So for a given context we predict a target word (probability) by training the shallow Neural Network. 
	
	![CBOW Architecture](https://i.stack.imgur.com/sAvR9.png)
 - **Skip-Gram** : The main principle here is almost **reverse** of CBOW i.e **given a target word we predict the context**.

<div align="center">
<img src="https://cdn-images-1.medium.com/max/800/1*SR6l59udY05_bUICAjb6-w.png" >
</div>

### 2. GloVe : Global Vectors for Word Representation

The main principle behind **global vectors is that we use the word-corpus to extract global one-word embedding i.e the embedding is global in sense with respect to the observed word corpus**.

It uses **word-frequency** and **co-occurence counts** to train the embedding model. It is an **unsupervised learning technique**.

<div align="center">
<img src="https://nlp.stanford.edu/projects/glove/images/comparative_superlative.jpg" >
<p><b>GloVe : Compartive Adjectives </b></p>
</div>


## Conclusion

In the next posts I will try explain these techniques in detail as well as implement them and evaluate their results.
