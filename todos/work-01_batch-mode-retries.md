This session is about the batch mode feature.
---

Actually when a task item is finished the batch mode engine move's to the next taskitem without checking the state of the finished one.

The state of the finished one should be that task item is now in the **todos/done** folder meaning
that the previous agent acknowledge the completeness of the task by moving it at this location.

We need to add a check:

- If that finished task is still in the **todos** folder (not moved to the todos/done folder) then:

	* Restart the task instead of moving to the next one
	* Add "Important: The task as been relaunch because *it has not been moved* to the **todos/done** folder (retry number *n*)" to the user message.

	this will make the user initial message (the one that starts the task) looking like this:

```
	You've been given task: todos/<task_name>.md to proceed. Do the coding tasks needed to implement the feature. Once done move it into the todos/done folder.

	Important: The task as been relaunch because *it has not been moved* to the **todos/done** folder (retry number *n*).
```

> n is the number of retries (from 1 to 3)

If **n** exceeds 3 then all the remaining tasks are aborted and the batch engine stops.

> This will give a chance for an agent to complete the unfinished task or just move it to the todos/done folder if he forgot it.

Possible reason for an unfinished task could be:

* Context overflow
* Connectivity error
* Tool use overflow
* Forgot to move to **todos/done** folder
