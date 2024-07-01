---
title : 'How a Python Decorator Crashed My Flask App: Lessons Learned'
## alternate title Python Decorators: How One Crashed My Flask App and the Lessons Learned
date : 2024-07-01T11:07:28+05:45
draft : true
tags : ["python"]
---

Python decorators are wonderful. They let you modify or extend the behavior of functions or methods without permanently modifying their source code. I've used them in several places in my code to add features to my functions. But recently, I came across a weird, or rather interesting, bug that I felt was worth sharing.
 
## The Issue

So I had a Flask API that was responsible for ML inference. Here is what the rough structure of the project looked like:

```
project/
├── app/
│   ├── utils/
│       ├── __init.py__
│       ├── decorators.py
|   ├── __init.py__
│   ├── app.py
│   ├── ...
```

`app/app.py`
```python
# all other imports
....
from utils.decorators import timeit

@app.route('/v1/predict/', methods=['POST'])
@timeit
def predict_v1(...):
    # do something
```

So as you can see above, I had a fairly simple decorator for timing the execution of an endpoint. Here is what a rough implementation of the decorator looked like: 

`app/decorators.py`

```python
import time

def timeit(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Function '{func.__name__}' executed in {elapsed_time:.4f} seconds")
        return result
    return wrapper
```

For those Python wizards out there who can immediately identify the issue by seeing the snippets above, feel free to skip the rest of this article (you guys have my respect). For those who do not see any issue with the above snippets, please read along. You deserve to know the issue! :)

This application was working fine for a long time until I decided to modify some pre-processing logic in the ML inference. I did not directly update the current endpoint as the changes were not quite backward compatible. So I created a `/v2/` endpoint for the new changes. My application now looked something like this:

`app/app.py`
```python
# all other imports
....
from utils.decorators import timeit

@app.route('/v1/predict/', methods=['POST'])
@timeit
def predict_v1(...):
    # do something

@app.route('/v2/predict/', methods=['POST'])
@timeit
def predict_v2(...):
    # do something, but differently ;)
```

Everything looked good, I had tested the new logic in isolation, and it was working exactly how I needed it to. So, with full confidence I started my flask server with `flask run` and the server instantly crashed.

Puzzled as to why this happened, I checked the logs. There, I saw a new error message. It wasn't the typical kind of message I was used to seeing at my job:

```bash
AssertionError: View function mapping is overwriting an existing endpoint function: wrapper
```

<div style="display: flex; justify-content: center; align-items: center;">
    <img src="/img/decorator/crash_meme.jpg" alt="crash meme lol">
</div>

## Understanding the Issue

Ok let's break the error message.

The first part of the message `AssertionError: View function mapping is overwriting an existing endpoint function` indicates that something went wrong in the Flask server regarding the route definitions. This message basically says that there are two routes with the same function name which is not allowed in Flask. You can refer to [this](https://dev.to/emma_donery/python-flask-app-routing-3l57) article to understand more about Flask routes.

But how did this happen? I had explicitly named the functions `predict_v1` and `predict_v2`. Why was this assertion error raised?

Ahh, the error message points to the culprit too: the function name `wrapper`.

After spending some time, I figured out what had happened: When we applied the `timeit` decorator to our functions, the decorator returned a new function named `wrapper`. This meant that both of our route functions `predict_v1` and `predict_v2`, were being replaced by a single function named `wrapper`. Flask uses the function name as the endpoint name by default. Since both decorated functions were now named `wrapper`, Flask tried to register multiple route functions with the same name, causing the `AssertionError`.

## The Solution

After understanding the issue, my immediate intuitive solution was to explicitly rename the wrapper function to the original function name inside the decorator before returning it.

`app/decorators.py`
```python
import time

def timeit(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Function '{func.__name__}' executed in {elapsed_time:.4f} seconds")
        return result
    wrapper.__name__ = func.__name__ # rename the function name before returning
    return wrapper
```




