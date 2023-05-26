# How to contribute to `plaquette`

Thanks for wanting to contribute! The suggestions in this page apply both to
code and documentation, so no need to read two things depending on what you
want to improve. :)

## Bugs

If you found a bug but have no solution for it:

1. **Look through the open issues** to see if this problem is not already
   reported.
2. If there's no sign of a matching issue, **open a new one**. The template
   that appears when you start a new issue should be enough guidance in how to
   write an effective bug report.
3. Don't bother to try and assign any metadata to the issue, someone will
   triage it as soon as possible.

## Fixes

If you found a bug *and* have a solution for it:

0. Thank you for sharing!
1. If the change is about more than a handful lines of code, please read our
   [developer guidelines](https://docs.plaquette.design/dev/index.html) to
   see what we expect in terms of code.
2. Open a new *draft* PR targeting `master` and make sure all checks pass
   (regression/literature tests will not start, that's expected while in
   draft-mode). If some checks do not pass, fix the offending part of the code
   (if you're unsure about how or what the check means, get in touch!);
   otherwise, switch from *draft* to *ready for review*;
3. Wait for somewhere to review the code and approve/suggest changes.
4. Done!

## Features

Depending on the feature you want to add, some planning might be required to
make sure everything aligns well together and most importantly that we are in
a position of being able to maintain the feature and fix it after you have
moved on to better things. It's always better to ask *first* if the feature you
want  to contribute is something we would consider adding.
