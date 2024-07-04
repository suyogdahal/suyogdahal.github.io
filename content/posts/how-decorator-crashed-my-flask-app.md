---
title : 'How a Decorator Crashed My Flask App: Lessons Learned'
## alternate title Python Decorators: How One Crashed My Flask App and the Lessons Learned
date : 2024-07-01T11:07:28+05:45
draft : false
tags : ["python"]
---

> TL;DR: Always use `functools.wraps` for your decorators.

Decorators in python are wonderful. They let you modify or extend the behavior of functions or methods without permanently modifying their source code. I've used them in several places in my code to add features to my functions. But recently, I came across a weird, or rather interesting, bug that I felt was worth sharing.

## The Issue

So we had a Flask API that was responsible for ML inference. Here is what the rough structure of the project looked like:

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

So as you can see above, we had a fairly simple decorator for timing the execution of an endpoint. Here is what a rough implementation of the decorator looked like: 

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

For those Python wizards out there who can immediately identify the issue by seeing the snippets above, feel free to skip the rest of this article (you guys have my respect). For those who didn't spot the issue from the snippets above, don't worry—you're in the right place! Let's walk through it together so you can understand the problem.

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

And to my surprise, this worked! (Ah, that feeling you get when your code works on the first try). But I wasn't satisfied with this solution. I wanted to understand it more. How could such a thing even exist? I am surely not the first programmer who has used decorators on their Flask routes. How could such an issue arise? I have been using decorators for over a couple of years, but why did this never occur to me? I was at the office, so I called my colleague [Ashish](https://asubedi.com.np/) to tell him about my newfound issue. Me being me, I just showed him the original decorator and asked him what's wrong here. He took some time, carefully read each and every line, and answered, "There's no `functools.wraps` in that decorator." 

How did I not see that? I have been using `functools.wraps` to create decorators, but really without understanding the why aspect of it. So, again I modified my decorator with `functools.wraps` and removed the function renaming part:

`app/decorators.py`
```python
import time
from functools import wraps


def timeit(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Function '{func.__name__}' executed in {elapsed_time:.4f} seconds")
        return result
    return wrapper
```

My Flask application did not crash this time!
Now I wanted to understand what exactly does `functools.wraps` do.

<div style="display: flex; justify-content: center; align-items: center;">
    <img src="/img/decorator/mind-blown.gif" alt="mind=blown">
</div>

## Understanding `functools.wraps`

Being the unsensible person I am, instead of searching the documentation of `functools.wraps`, I directly opened the source code of it.

```python
def wraps(wrapped,
          assigned = WRAPPER_ASSIGNMENTS,
          updated = WRAPPER_UPDATES):
    """Decorator factory to apply update_wrapper() to a wrapper function

       Returns a decorator that invokes update_wrapper() with the decorated
       function as the wrapper argument and the arguments to wraps() as the
       remaining arguments. Default arguments are as for update_wrapper().
       This is a convenience function to simplify applying partial() to
       update_wrapper().
    """
    return partial(update_wrapper, wrapped=wrapped,
                   assigned=assigned, updated=updated)
```

It returns a partial of the function `update_wrapper`. There was nothing I could make out of this function alone. So I moved to the implementation of `update_wrapper`. This is where things got interesting.

```python
WRAPPER_ASSIGNMENTS = ('__module__', '__name__', '__qualname__', '__doc__',
                       '__annotations__', '__type_params__')
WRAPPER_UPDATES = ('__dict__',)
def update_wrapper(wrapper,
                   wrapped,
                   assigned = WRAPPER_ASSIGNMENTS,
                   updated = WRAPPER_UPDATES):
    """Update a wrapper function to look like the wrapped function

       wrapper is the function to be updated
       wrapped is the original function
       assigned is a tuple naming the attributes assigned directly
       from the wrapped function to the wrapper function (defaults to
       functools.WRAPPER_ASSIGNMENTS)
       updated is a tuple naming the attributes of the wrapper that
       are updated with the corresponding attribute from the wrapped
       function (defaults to functools.WRAPPER_UPDATES)
    """
    for attr in assigned:
        try:
            value = getattr(wrapped, attr)
        except AttributeError:
            pass
        else:
            setattr(wrapper, attr, value)
    for attr in updated:
        getattr(wrapper, attr).update(getattr(wrapped, attr, {}))
    # Issue #17482: set __wrapped__ last so we don't inadvertently copy it
    # from the wrapped function when updating __dict__
    wrapper.__wrapped__ = wrapped
    # Return the wrapper so this can be used as a decorator via partial()
    return wrapper
```

- `update_wrapper` updates a wrapper function to look like the wrapped function.
- It copies the `__module__`, `__name__`, `__qualname__`, `__doc__`, and `__annotations__` attributes from the wrapped function to the wrapper function. This default list is defined in `WRAPPER_ASSIGNMENTS`.
- It updates the `__dict__` of the wrapper with all elements from the wrapped function's `__dict__`, based on `WRAPPER_UPDATES`.
- It sets a new `__wrapped__` attribute on the wrapper, pointing to the original function.

In a nutshell, `functools.wrap` ensures that the decorated function retains the original function's signature, documentation, and other attributes. (So bascially an extented version of my intial solution 😅).

In conclusion, encountering issues like this serves as a valuable reminder of the subtle complexities and hidden layers that lie beneath the surface of programming. It shows that no matter how much we learn, there's always more to discover! Keep learning, cheers 🥂.