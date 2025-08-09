from edql_base_env import EDQLBaseEnv

class EDQLChandniChowkEnv(EDQLBaseEnv):
    def __init__(self, net_file, route_file, use_gui=False, max_steps=1000):
        super().__init__(net_file, route_file, use_gui=use_gui, max_steps=max_steps)
        try:
            self.compute_edge_successors()
        except Exception:
            pass
