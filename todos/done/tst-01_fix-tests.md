There is some errors in several tests modules within tests folder.

These tests are all green on Linux so they might be related to the current process to move to a cross platform (Linux / windows) code base.

Take modules one by one and when you find a failing test case.

* Find what is the issue (in code or in the test case itself).
* Fix it
* Move on to the next module.

Do not try to run the all test suite since it takes a long time.

> Tests need to be run using the virtualenv located in **.\.venv\Scripts** folder

* If a test module hangs, run test case one by one instead of increasing time out.