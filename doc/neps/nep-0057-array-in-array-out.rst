==============================
NEP 57 — Array In -> Array Out
==============================

:Author: Matt Haberland <mhaberla@calpoly.edu>, Add Your Name Here <example@example.com>
:Status: Draft
:Type: Standards Track
:Created: 2025-05-14
:Resolution:

Abstract
--------

:ref:`NEP56` proposed adding nearly full support for the array API standard,
but many operations involving higher rank arrays still return scalars instead
of zero-rank arrays. This NEP would redefine the result of these operations
to be zero-rank arrays.

Motivation and scope
--------------------

The 2024.12 version of the array API standard [1]_ states:

   Apart from array object attributes, such as ``ndim``, ``device``, and
   ``dtype``, all operations in this standard return arrays (or tuples of
   arrays)...

Beginning with :ref:`NEP56` and NumPy 2.0.0, NumPy added nearly full support
for the standard, but explicitly deferred compliance with this aspect.

   We note that one NumPy-specific behavior that remains is returning array
   scalars rather than 0-D arrays in most cases where the standard, and other
   array libraries, return 0-D arrays (e.g., indexing and reductions)...
   There have been multiple discussions over the past year about the
   feasibility of removing array scalars from NumPy, or at least no longer
   returning them by default. However, this would be a large effort with some
   uncertainty about technical risks and impact of the change, and no one has
   taken it on.

This NEP represents an effort to "take it on". It is a worthwile undertaking:
scalars "basically duck type 0-D arrays", but they do not *fully* duck type
zero-rank arrays, with the most fundamental difference being that scalars are
immutable and zero-rank arrays are not.

It may be argued that if instances of array base class ``np.ndarray`` and
scalar base class ``np.generic`` were fully interoperable, together, they
would implement a protocol compatible with the array API standard. Even if this
were the case, this design is complex and leads to confusion and errors due to
self-inconsistency (zero-rank array-like scalars are immutable, but arrays of
other rank are mutable) and inconsistency with all other array API compatible
libraries. In particular, it leads to difficulties in working with vectorized
reducing functions, which begin with arrays of rank :math:`N` and return
objects of rank :math:`M < N`: when :math:`M = 0`, the rules change. This
prompts an unfortunate pattern of calling ``asarray`` on the results of
intermediate array operations to ensure that operations like boolean mask
assignment still work. The inconsistency also presents downstream library
authors with an unfortunate choice: should they maintain consistency with
NumPy and prefer to return scalars when possible (e.g. ``scipy.stats``, which
explicitly uses empty-tuple indexing on all results to *ensure* consistency,
and ``scipy.special``, which relies on NumPy ufunc machinery), or should they
follow the lead of the array API standard and prefer zero-rank arrays (e.g.
``scipy.interpolate``).

Usage and impact
----------------

Currently, most operations in NumPy involving zero-rank arrays return scalars,
reducing operations that would naturally result in a zero-rank array actually
produce a scalar, and indexing operations that would naturally result in a
zero-rank array actually produce a scalar. The proposal is for these operations
to return zero-rank arrays instead of scalars.

.. code:: python

   import numpy as np
   x = np.asarray(1.)
   np.isscalar(x + x)  # True (main), False (NEP)
   np.isscalar(np.exp(x))  # True (main), False (NEP)
   y = np.ones(10)
   np.isscalar(np.sum(y), axis=-1))  # True (main), False (NEP)
   np.isscalar(y[0])  # True (main), False (NEP)

For exceptions to these rules, ask Sebastian.

Empty-tuple indexing may still be used to cast any resulting zero-rank array
to the corresponding scalar.

.. code:: python

   import numpy as np
   x = np.asarray(1.)
   np.isscalar(x + x)  # True (main), False (NEP)

The main impact to users is more predictable results due to improved
consistency within NumPy, between NumPy and the array API standard, and
between NumPy and other array libraries. Working with the results of reducing
functions, in particular, will be easier because return values of any rank
will support boolean indexing assignment.

There is a secondary impact on performance. On typical hardware, execution
time of conversion from zero-rank arrays to scalars and elementary arithmetic
operations involving only scalars is on the order of tens of nanoseconds,
whereas operations involving only zero-rank arrays is on the order of hundreds
of nanoseconds. Consequently, some elementary arithmetic calculations will be
slower. On the other hand, conversion from scalars to zero-rank arrays takes a
few hundred nanoseconds, and many operations, such such as ufuncs and
operations involving both scalars and rank-zero arrays require conversion from
scalars to zero-rank arrays. These operations will be faster. We will not
speculate as to whether this will have a net positive or net negative impact on
user applications, but the net impact is expected to be small since impact on
downstream library test suites has been minimal in testing.

Backward compatibility
----------------------

The motivation of this proposal is to eliminate the surprises associated with
a scalar result when a zero-rank array would be expected. However, existing
code may rely on the current behavior, and this presents backward compatibility
concerns.

The main concern for user code is that the mutable zero-rank arrays that
replace immutable scalars are no longer hashable. For instance, they cannot
by used directly as keys of dictionaries, the argument of an ``lru_cache``
-decorated function, etc. In all circumstances, tbe patch is simple: convert
the zero-rank array to a scalar with empty-tuple indexing.

Running the test suites of dependent libraries against a branch of NumPy that
implements these changes has revealed a few other issues. <Library maintainers,
please summaries these issues here.>

Detailed description
--------------------

The new functionality will be used in much the same was as old functionality,
except that sometimes authors will need to convert zero-rank arrays to scalars
rather than converting scalars to zero-rank arrays. For instance, consider the
following:

.. code:: python

   import numpy as np
   rng = np.random.default_rng(85878653462722874072976519960992129768)
   x = rng.standard_normal(size=10)
   y = np.sum(x, axis=-1)
   z = {y: 'a duck'}  # use scalar result as dictionary key
   y = np.asarray(y)  # convert to array to allow mutation
   y[y < 0] = np.nan

The use of ``x`` as a dictionary key would need to become ``x[()]``, but ``z``
no longer needs to be explicitly converted to an array.

.. code:: python

   z = {y[()]: 'a duck'}  # extract scalar to use as dictionary key
   y[y < 0] = np.nan      # no conversion to array required

Realistic examples can be found throughout the codebases of dependent
libraries.

Related work
------------

All known libraries that attempt to implement the array API standard
(e.g. ``cupy``, ``torch``, ``jax.numpy``, ``dask.array``) return
zero-rank arrays as specified by the standard.

Implementation
--------------

To implement the NEP, the branch prepared by Sebastian needs to be merged,
and dependent libraries will need to prepare releases that adapt to (and take
advantage of) the changes. Branches of libraries including SciPy and Matplotlib
have already been prepared without much difficulty. Initially, users will have
the option of opting into this behavior using an environment variable, so these
releases need to be compatible with both old and new behaviors. To make the new
behavior the default and only behavior, NumPy will need to advise users of the
pending change and give the appropriate notice. The initial draft of this
document does not specify the appropriate timeline and procedures; this line
will be updated to reflect the consensus of the maintainer team.

Alternatives
------------

There are two main alternatives to this proposal.

The alternative suggested by :ref:`NEP56` is to maintain the current behavior
and make NumPy scalars more fully duck-type zero-rank arrays, such as adding
missing behaviors.

While this work would still be valuable, we propose the behavior change to more
fully comply with the standard and to resolve the problems mentioned in the
Motivation.

More extreme alternatives are also available, such as eliminating NumPy
scalars entirely. We do not take this approach for two reasons:

1. Scalars still have some advantages as hashable, dtyped objects that support
   very fast elementary arithmetic.
2. The backward compatibility concerns with eliminating scalars entirely are
   much more severe.

A variant of this proposal is to eliminate the exceptional behavior associated
with reducing operations and ``axis=None``. I cannot present any justification
for why we shouldn't do this; I would certainly prefer it because it would be
even more consistent and fully compliant with the standard. Ask Sebastian.

Discussion
----------

This section may just be a bullet list including links to any discussions
regarding the NEP:

- https://github.com/numpy/numpy/issues/24897
- https://github.com/scientific-python/summit-2025/issues/38
- https://github.com/scipy/scipy/pull/22947#discussion_r2080108060


References and footnotes
------------------------

.. [1] `Python array API standard 2014.12 — Array Object`_
.. [2] Each NEP must either be explicitly labeled as placed in the public domain (see
   this NEP as an example) or licensed under the `Open Publication License`_.

.. _Open Publication License: https://www.opencontent.org/openpub/
.. _Python array API standard 2014.12 — Array Object: https://data-apis.org/array-api/latest/API_specification/array_object.html

Copyright
---------

This document has been placed in the public domain. [2]_
