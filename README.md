# Image-Recognision

## How to make an AI able to detect images?
In fact a quite simple task:

instead of trying to make the AI analyze the full image (say an 1024x1024 or whatever image), we can divide the image into smaller sections - like say 5x5 images or so on. 
Now, in each of these smaller images, what we can do is just calculate the average RGB value of the pixels. 
Repeat this process for each of the smaller pixel clusters.

For each image we manually feed to the AI and we manually define what it is meant to be, the AI slowly learns the rough patterns of that image. 
How? Well - as we manually tell the AI that image 1 is a *dog*, the AI is specifically taught that cluster X containing an average Red color of say 50% means being a dog. 
We can feed the AI thus hundreds, or even thousands, of images of the same concept - the AI will slowly tend to learn patterns by generalizing / averaging the each individual RGB color in each of the clusters. 

But what do we even do with the cluster values?
Well obviously we take the data from the clusters, and send that signal to the image recognition layer and it guesses how likely it is that the appropriate image is in fact concept A by comparing the given values of this image to the historical / trained ones. 
the one with the closest values is picked as the one that the AI displays as being recognized, i.e. - if say the aggregated RGB colors match 90% to that of an *dog*, the AI displays with 90% confidence it being a dog.

But what about color?
An massive flaw and problem we run into is - what if say we train the AI on red traffic signs and we will one day give it an blue traffic sign to analyze: it will fail miserably even if the image is just an Photoshop color change operation. why? because it learned
that the COLOR in that specific pattern is the traffic sign. How to fix this? in fact its really simple to fix it - just add an additional / parallel model that is identical to this one, just one that is only fed gray-scale versions of the same images. this one quite by
definition and nature wont be able to learn that *red* is the color it should be looking for. When evaluating an image, simply pick the output with the highest confidence: if the RGB model shows higher confidence, pick its choice, if not - then pick the gray-scale one's.
Thus, the AI has learned GENERALIZATION - it is able to generalize what an traffic sign is able to look like, no matter the color. 

But what about patterns?
say - a zebra. the AI will of-course just average out the colors, thus - unable to actually learn what a zebra looks like because it always takes the aggregate RGB colors. How to fix this? well - lets add an additional separate layer: one that specifically looks out for stripped / line patterns. To incorporate it into the wider model, we have to make it an *consultant* - it modifies the output of the others by giving its own confidence feedback. 

Can we generalize the patterns / add more patterns?
Yes - there is no theoretical limit. The only practical limitations are: computational ability, complexity of introduction / maintenance, training data volume. frontier models often include tens and even hundreds of such secretary layers.

How do we compare / measure an AI model like such?
Mostly in 2 values: 
1. quality (% of guesses that are correct)
2. quantity (the amount of things its able to categorize)

So measuring an AI is like deciding between Volts (quality) or Amps (Quantity) - its best to pick both: Pick Watts. 


How do we tell the AI what to actually categorize?

At first, we have to manually define it - we have to manually give the AI specifically manually categorized items (e.g. - 10 pictures of *tractor*) and the AI will slowly learn to categorize tractors. But if we are looking extra fancy, we can even introduce an quite special
technique of unsupervised training / categorizing. Basically - the AI is of-course looking at countless images, and it itself is able to detect patterns already, so what if it tries to categorize its own patterns?

How does this self-categorization work?

The AI say takes a random image - then it compares random parts of this image to countless others images and generalizes specific parts, then it labels this thing as say *category x* - afterwards a human is required to manually check and verify if category x is a valid pattern that in fact does exist as an item to recognize, or its an undefined pattern, at which point it will probably be just deleted / forgotten. 

How do we test the AIs actual ability to detect the specific part of the image and not just guess based on the surrounding context? e.g. - detect a dog not by the fact that its standing next to an dog bed, but because its an dog. 
quite simple - take the image of the dog near the dog bed and digitally alter the image to remove / replace parts of the image, such as - removing the head, body parts, the dog bed and so on. Compare the model's output guess in all of these, to figure out which one the AI is actually basing its guess on: if blacking out the dog's head severely decreases the confidence / accuracy, the AI is clearly basing its guess off of the ability to recognize an dog's head. 

A thought: what if instead of making fixed 5x5 pixel image clusters, we generated random variations of them (such as with the use of Voronoi diagrams and similar)? What if we made the cluster be sliding and not constant, i.e. - a sweeping cluster? 

What is also fascinating about the image recognition AI is that most of the systems and ideas / principles used here are also the exact same ones employed by humans and other animals in their own vision. 
