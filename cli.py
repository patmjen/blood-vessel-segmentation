class TrackArgActionWrapper(object):
    """
    Wraps an argparse.Action
    """
    def __init__(self, action, extra_name):
        # Need to use __dict__ to avoid going through __setattr__.
        # Afterwards, we can use the names normally, since __getattr__ is only
        # called if the requested attribute cannot be found.
        self.__dict__['action'] = action
        self.__dict__['extra_name'] = extra_name


    def __call__(self, parser, namespace, values, option_string=None):
        if not hasattr(namespace, self.extra_name):
            setattr(namespace, self.extra_name, set())

        getattr(namespace, self.extra_name).add(self.action.dest)
        self.action(parser, namespace, values, option_string)


    def __getattr__(self, item):
        return getattr(self.action, item)


    def __setattr__(self, item, value):
        setattr(self.action, item, value)


def add_argument_tracking(parser, extra_name='seen_args_'):
    """
    Track which arguments were actually given to ArgumentParser.

    If any arguments get a non-default value their names are stored in an extra
    set in the Namespace returned by parser.parse_args. This allows one to
    determine if a value was specified by the user or if it got it's default
    value.

    Parameters
    ----------
    parser
        ArgumentParser object to modify
    extra_name
        Name of set with user given arguments. Default: seen_args_

    Returns
    -------
    ArgumentParser
        Modified parser object

    Example
    -------
    from argparse import ArgumentParser
    from cli import add_argument_tracking

    parser = ArgumentParser()
    parser.add_argument('--foo', default=1, type=int)
    parser.add_argument('--bar', default=2, type=int)
    parser = add_argument_tracking(parser)

    args = parser.parse_args(['--foo', '1'])
    print('foo =', args.foo)                # foo = 1
    print('bar =', args.bar)                # bar = 2
    print('seen_args_ =', args.seen_args_)  # seen_args_ = {'foo'}
    """
    # NOTE: We're reaching into the bowels of ArgumentParser here. This may
    # break in other Python releases!
    parser._actions = [TrackArgActionWrapper(a, extra_name)
                       for a in parser._actions]
    parser._option_string_actions = { k: TrackArgActionWrapper(a, extra_name)
        for k, a in parser._option_string_actions.items() }
    return parser
