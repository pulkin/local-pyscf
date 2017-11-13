import os
import inspect


class LogCalls(object):
    def __init__(self, name, head="", line_start="", tail=""):
        """A simple facility logging all method calls."""
        self.name = name
        if os.path.exists(name):
            os.remove(name)
        self.items = []
        self.head = head
        self.line_start = line_start
        self.tail = tail

    def __getattr__(self, item):
        def __(*args, **kwargs):
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
                            ([", ".join(repr(j) for j in i["__args__"])] if len(i["__args__"]) > 0 else []) +\
                            ([", ".join(j+"="+repr(k) for j, k in i["__kwargs__"].items())] if len(i["__kwargs__"]) > 0 else [])
                        )
                    ) for i in self.items),
                    self.tail,
                ]))
        return __


def pyplot(name=None):
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
    )