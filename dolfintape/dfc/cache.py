# Copyright (C) 2016 Jan Blechta
#
# This file is part of dolfin-tape.
#
# dolfin-tape is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# dolfin-tape is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with dolfin-tape. If not, see <http://www.gnu.org/licenses/>.

from __future__ import print_function
from pprint import pprint
import os

__all__ = ["clear", "stats"]

_cache = {}
_stats = {"load hits": 0, "load miss": 0, "stores done": 0, "stores ignored": 0,
          "overwrites approved": 0, "overwrites denied": 0}

def load(key):
    # Fetch from cache and return or raise
    try:
        value = _cache[key]
    except KeyError as e:
        _stats["load miss"] += 1
        raise e
    else:
        _stats["load hits"] += 1
        return value

def store(key, value, allow_overwrite=False):
    ## Caching disabled for some reason
    #if ignore:
    #    _stats["stores ignored"] += 1
    #    return value

    # Handle overwrites (typically should not happen)
    if key in _cache:
        if not allow_overwrite:
            _stats["ovewrites denied"] += 1
            raise AssertionError("Attempt to store key '%s' to compiler cache again!" % key)
        else:
            _stats["ovewrites approved"] += 1

    # Store and return back value
    _cache[key] = value
    _stats["stores done"] += 1
    return value

def clear(clear_stats=True, clear_cache=True):
    if clear_stats:
        _cache.update(dict.fromkeys(_cache, 0))
    if clear_cache:
        _cache.clear()

def stats():
    s = _stats.copy()
    s["items in cache"] = len(_cache)
    return s

def list_stats(clear_stats=False, clear_cache=False):
    print(80*"="+os.linesep+"DFC stats:"+os.linesep+80*"-")
    pprint(stats())
    print(80*"=")
    clear(clear_stats, clear_cache)
