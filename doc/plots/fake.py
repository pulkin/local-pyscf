import os
import inspect
import numpy
import json


def p(x):
    if isinstance(x, numpy.ndarray):
        x = x.tolist()
    return json.dumps(x)


class LogCalls(object):
    def __init__(self, name, head="", line_start="", tail="", proto=None):
        """A simple facility logging all method calls."""
        self.name = name
        if os.path.exists(name):
            os.remove(name)
        self.items = []
        self.head = head
        self.line_start = line_start
        self.tail = tail
        self.proto = proto

    def __getattr__(self, item):
        def __(*args, **kwargs):
            if self.proto is not None:
                getattr(self.proto, item)(*args, **kwargs)
            self.items.append(
                dict(
                    __method_name__=item,
                    __args__=args,
                    __kwargs__=kwargs,
                )
            )
            with open(self.name, 'w') as f:
                f.write("\n".join([
                    self.head,
                    "\n".join("{line_start}{caller}({arguments})".format(
                        line_start=self.line_start,
                        caller=i["__method_name__"],
                        arguments=", ".join(
                            ([", ".join(p(j) for j in i["__args__"])] if len(i["__args__"]) > 0 else []) +\
                            ([", ".join(j+"="+p(k) for j, k in i["__kwargs__"].items())] if len(i["__kwargs__"]) > 0 else [])
                        )
                    ) for i in self.items),
                    self.tail,
                ]))
        return __


def pyplot(name=None):
    from matplotlib import pyplot
    if name is None:
        name = inspect.stack()[1][1]
        if name[-3:] == ".py":
            name = name[:-3]
        name = name+"_cached.py"
    return LogCalls(
        name,
        head="#/usr/bin/env python\n\"\"\"\nThis file was generated automatically.\n\"\"\"\nfrom matplotlib import pyplot",
        line_start="pyplot.",
        tail="\n",
        proto=pyplot,
    )