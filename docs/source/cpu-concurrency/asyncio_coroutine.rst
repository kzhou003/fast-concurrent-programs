Asyncio Coroutine - Finite State Machine

Overview
--------
This script demonstrates asyncio coroutines by implementing a finite state machine (FSM) that randomly transitions between states until reaching an end state.

File Location
``basics/08*asyncio_coroutine.py``

Key Concepts

Finite State Machine (FSM)
~~~~~~~~~~~~~~~~~~~~~~~~~~
A computational model consisting of states and transitions between those states. The system is always in exactly one state at a time.

Coroutines
~~~~~~~~~~
Special functions defined with ``async def`` that can pause execution and yield control back to the event loop using ``await``.

State Transitions
~~~~~~~~~~~~~~~~~
The FSM randomly chooses the next state based on a random input value (0 or 1), simulating decision-making logic.

Code Breakdown

State Machine Structure
~~~~~~~~~~~~~~~~~~~~~~~
::

Start State
    down

State 3 -+--> State 1
         +--> End State
::


Start State
~~~~~~~~~~~
.. code-block:: python

async def start_state():
    print('Start State called\n')
    input_value = randint(0, 1)
    await asyncio.sleep(1)

    if input_value == 0:
        result = await state2(input_value)
    else:
        result = await state1(input_value)

    print(f'Resume of the Transition : \nStart State calling {result}')
::


Entry point of the FSM that randomly transitions to State 1 or State 2.

Intermediate States (State 1, 2, 3)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. code-block:: python

async def state1(transition_value):
    output_value = f'State 1 with transition value = {transition_value}\n'
    input_value = randint(0, 1)
    await asyncio.sleep(1)

    print('...evaluating...')
    if input_value == 0:
        result = await state3(input_value)
    else:
        result = await state2(input_value)

    return output_value + f'State 1 calling {result}'
::


Each state:
1. Records its transition value
2. Waits asynchronously (simulates processing)
3. Randomly chooses the next state
4. Returns a string tracking the transition path

End State
~~~~~~~~~
.. code-block:: python

async def end_state(transition_value):
    output_value = f'End State with transition value = {transition_value}\n'
    print('...stop computation...')
    return output_value
::


Terminal state that ends the computation and returns the final result.

Python 3.12 Updates

Changes Made
~~~~~~~~~~~~
1. **Replaced ``@asyncio.coroutine`` decorator with ``async def``**
   - Old: ``@asyncio.coroutine`` + ``def function()``
   - New: ``async def function()``
   - The decorator was deprecated in Python 3.8

2. **Replaced ``yield from`` with ``await``**
   - Old: ``result = yield from state2(input_value)``
   - New: ``result = await state2(input_value)``
   - ``await`` is the modern, cleaner syntax

3. **Replaced ``time.sleep()`` with ``await asyncio.sleep()``**
   - Prevents blocking the event loop
   - Allows proper asynchronous execution

4. **Updated to ``asyncio.run()``**
   - Old: ``loop = asyncio.get_event*loop(); loop.run_until*complete()``
   - New: ``asyncio.run(start_state())``
   - Simpler and handles cleanup automatically

5. **Modernized string formatting**
   - Replaced ``%`` formatting with f-strings
   - More readable and Pythonic

Execution Flow Example

::

start_state() -> (random: 1)
  down
state1(1) -> (random: 0)
  down
state3(0) -> (random: 1)
  down
end_state(1)
  down
Returns complete transition path
::


Usage
-----
.. code-block:: bash

python3 08*asyncio_coroutine.py
::


Output Example
::

Finite State Machine simulation with Asyncio Coroutine
Start State called

...evaluating...
...evaluating...
...evaluating...
...stop computation...
Resume of the Transition :
Start State calling State 2 with transition value = 0
State 2 calling State 1 with transition value = 0
State 1 calling State 3 with transition value = 0
State 3 calling End State with transition value = 1
::


Key Takeaways

1. **Async Recursion**: Coroutines can call other coroutines recursively using ``await``
2. **Return Values**: Async functions can return values just like regular functions
3. **State Pattern**: FSMs are naturally expressed with async/await
4. **Path Tracking**: String concatenation allows tracking the execution path through states
5. **Non-deterministic**: Random transitions create different paths each run

Use Cases
---------

This pattern is useful for:
- Workflow engines
- Game state management
- Protocol implementations
- Decision trees
- Process automation
