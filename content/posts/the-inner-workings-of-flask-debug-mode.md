---
title: 'The Inner Workings of Flask Debug Mode'
date: 2024-11-08T22:08:03-05:00
draft: true
---

I didn’t expect that in an attempt to save a few seconds of manual labor, I would end up diving deep into the inner workings of Flask—and its engine, Werkzeug. This article is an attempt to document my findings and learnings along the way.

I was working on a real-time object detection project that began as a simple Python script. But soon, the requirement grew to display some of the result in a webpage.

Since it was an experimental project, I came up with a hacky solution to meet the requirement: periodically save the detection results to a JSON file and build a small Flask app to periodically read the data and show it on a web app. Here is what my workflow looked like:

1. Run the object detection script: `python obj_detect.py`
2. Periodically save the output to a JSON file
3. Run the Flask app: python app.py, which reads from the JSON file and updates the web display.

Now, with this setup, every time I needed to start the project, I had to run two commands for detection and one for the app. But being the impatient engineer I am, running two commands to start an application felt like a waste of time—why do something manually in 5 seconds when you can spend your whole day automating it? 🤷
