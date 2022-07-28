from . import ltldiff as ltd
import torch
from .domains import Box
import numpy as np

def constraint_loss(constraint, ins, targets, zs, net, rollout_func):
    cond = constraint.condition(zs, ins, targets, net, rollout_func)
    
#     neg_losses = ltd.Negate(cond).loss(0)
    losses = cond.loss(0)
    sat = cond.satisfy(0)
    neg_losses = torch.zeros(sat.shape)

        
    return neg_losses, losses, sat

def fully_global_ins(ins, epsilon):
    low_ins = ins - epsilon
    high_ins = ins + epsilon
    return [Box(low_ins[:, i], high_ins[:, i]) for i in range(ins.shape[1])]


class EventuallyReach:
    def __init__(self, reach_ids, epsilon):
        # The index of object in start info that you need to eventually reach
        self.reach_ids = reach_ids 
        self.epsilon = epsilon

    def condition(self, zs, ins, targets, net, rollout_func):
        weights = net(zs)
        rollout_traj = rollout_func(zs[:, 0], zs[:, -1], weights)[0]

        rollout_term = ltd.TermDynamic(rollout_traj)

        reach_constraints = []
        for reach_id in self.reach_ids:
            reach_point_batch = zs[:, reach_id]
            reach_constraints.append(ltd.Eventually(
                ltd.EQ(rollout_term, ltd.TermStatic(reach_point_batch)),
                rollout_traj.shape[1]
            ))
        
        return ltd.And(reach_constraints)

    def domains(self, ins, targets):
        return fully_global_ins(ins, self.epsilon)

class StayInZone:
    def __init__(self, min_bound, max_bound, epsilon):
        assert min_bound.dim() == 1, "Num of dims min bound should be 1: (Spatial Dims)"
        assert max_bound.dim() == 1, "Num of dims for max bound should be 1: (Spatial Dims)"

        self.min_bound = min_bound
        self.max_bound = max_bound
        self.epsilon = epsilon

    def condition(self, zs, ins, targets, net, rollout_func):
        weights = net(zs)
        rollout_traj = rollout_func(zs[:, 0], zs[:, -1], weights)[0]
        rollout_term = ltd.TermDynamic(rollout_traj)

        return ltd.Always(
            ltd.And([
                ltd.GEQ(rollout_term, ltd.TermStatic(self.min_bound)),
                ltd.LEQ(rollout_term, ltd.TermStatic(self.max_bound))
            ]),
            rollout_traj.shape[1]
        )

    def domains(self, ins, targets):
        return fully_global_ins(ins, self.epsilon)


class LipchitzContinuous:
    def __init__(self, smooth_thresh, epsilon):
        self.smooth_thresh = smooth_thresh
        self.epsilon = epsilon

    def condition(self, zs, ins, targets, net, rollout_func):
        weights_x = net(ins)
        weights_z = net(zs)

        rollout_x = rollout_func(ins[:, 0], ins[:, -1], weights_x)[0]
        rollout_z = rollout_func(zs[:, 0], zs[:, -1], weights_z)[0]

        rollout_diffs = ltd.TermDynamic(torch.abs(rollout_x - rollout_z))
        input_diffs = ltd.TermStatic(self.smooth_thresh * torch.abs(ins - zs))

        return ltd.Always(
            ltd.LEQ(rollout_diffs, input_diffs),
            rollout_x.shape[1]
        )

    def domains(self, ins, targets):
        return fully_global_ins(ins, self.epsilon)


class DontTipEarly:
    def __init__(self, orientation_lb, orientation_ub, tip_id, min_dist, epsilon):
        self.tip_id = tip_id
        self.min_dist = min_dist
        self.orientation_lb = orientation_lb
        self.orientation_ub = orientation_ub
        self.epsilon = epsilon

    def condition(self, zs, ins, targets, net, rollout_func):
        weights = net(zs)
        rollout = rollout_func(zs[:, 0], zs[:, -1], weights)[0]

        # Assuming xyz-rpy
        rotation_terms = ltd.TermDynamic(rollout[:, :, 3:])

        # Measuring the distance across dimensions...
        tip_point = zs[:, self.tip_id, 0:2]

        dist_to_tip = ltd.TermDynamic(torch.norm(rollout[:, :, 0:2] - tip_point[:, None, :], dim=2, keepdim=True))

        return ltd.Always(
            ltd.Implication(
                ltd.Or([
                    ltd.GEQ(dist_to_tip, ltd.TermStatic(self.min_dist)),
                    ltd.LEQ(ltd.TermDynamic(rollout[:, :, 2:3]), ltd.TermStatic(zs[:, 1, 2:3]))
                ]),
                ltd.And([
                    ltd.GEQ(rotation_terms, ltd.TermStatic(self.orientation_lb)),
                    ltd.LEQ(rotation_terms, ltd.TermStatic(self.orientation_ub))
                ])
            ),
            rollout.shape[1]
        )

    def domains(self, ins, targets):
        return fully_global_ins(ins, self.epsilon)



class MoveSlowly:
    def __init__(self, max_velocity, epsilon):
        self.max_velocity = max_velocity
        self.epsilon = epsilon

    def condition(self, zs, ins, targets, net, rollout_func):
        weights = net(zs)
        rollout = rollout_func(zs[:, 0], zs[:, -1], weights)[0]

        displacements = torch.zeros_like(rollout)
        displacements[:, 1:, :] = rollout[:, 1:, :] - rollout[:, :-1, :] # i.e., v_t = x_{t + 1} - x_t
        velocities = ltd.TermDynamic(torch.norm(displacements, dim=2, keepdim=True))

        return ltd.Always(
            ltd.LT(velocities, ltd.TermStatic(self.max_velocity)),
            rollout.shape[1]
        )

    def domains(self, ins, targets):
        return fully_global_ins(ins, self.epsilon)


class AvoidPoint:
    def __init__(self, point_id, min_dist, epsilon):
        self.point_id = point_id
        self.min_dist = min_dist
        self.epsilon = epsilon

    def condition(self, zs, ins, targets, net, rollout_func):
        weights = net(zs)
        rollout = rollout_func(zs[:, 0], zs[:, -1], weights)[0]

        point = zs[:, self.point_id]

        dist_to_point = ltd.TermDynamic(
            torch.norm(rollout - point[:, None, :], dim=2, keepdim=True)
        )

        return ltd.Always(
            ltd.GT(dist_to_point, ltd.TermStatic(self.min_dist)),
            rollout.shape[1]
        )

    def domains(self, ins, targets):
        return fully_global_ins(ins, self.epsilon)

class AvoidAndPatrolConstant:
    def __init__(self, av_point, patrol_point, min_avoid_dist, epsilon):
        self.av_point = av_point
        self.patrol_point = patrol_point
        self.epsilon = epsilon
        self.min_avoid_dist = min_avoid_dist

    def condition(self, zs, ins, targets, net, rollout_func):
        weights = net(zs)
        rollout = rollout_func(zs[:, 0], zs[:, -1], weights)[0]


        dist_avoid = ltd.TermDynamic(
            torch.norm(rollout[:, :, 0:2] - self.av_point, dim=2, keepdim=True)
        )

        return ltd.And([
            ltd.Always(
                ltd.GT(dist_avoid, ltd.TermStatic(self.min_avoid_dist)),
                rollout.shape[1]
            ),
            ltd.Eventually(
                ltd.EQ(ltd.TermDynamic(rollout[:, :, 0:2]), ltd.TermStatic(self.patrol_point)),
                rollout.shape[1]
            )
        ])

    def domains(self, ins, targets):
        return fully_global_ins(ins, self.epsilon)
    
class SkillConstraint:
    """
    This will create the constraints on a trajectory to obey the skill.

    Constraint:


    """
    def __init__(self, symbols, suggestions_intermediate_all_pres, suggestions_intermediate_all_posts, suggestion_post, suggestion_unique, hard_constraints,
                 workspace_bnds, epsilon, opts):
        self.symbols = symbols
        self.suggestions_intermediate_all_pres = suggestions_intermediate_all_pres
        self.suggestions_intermediate_all_posts = suggestions_intermediate_all_posts
        self.suggestion_post = suggestion_post
        self.suggestion_unique = suggestion_unique
        self.hard_constraints = hard_constraints
        self.epsilon = epsilon
        self.workspace_bnds = workspace_bnds
        self.opts = opts

    def condition(self, zs, ins, targets, net, rollout_func):
        weights = net(zs)

        # The actual trajectory
        rollout_traj = rollout_func(zs[:, 0], zs[:, -1], weights)[0]
        rollout_term = ltd.TermDynamic(rollout_traj)

        # Each symbol corresponds to a constraint
        sym_ltd, neg_sym_ltd = create_sym_and_neg_ltd_dicts(self.symbols, rollout_term)

        # A postcondition or the same precondition will always come after a given precondition
        # pre -> N (pre | posts) is the same as !pre | N (pre | posts)
        implication_next_list = []
        for suggestion_intermediate in self.suggestions_intermediate_all_posts:
            # A post will come after the pres
            pre_ltd = ltd.Next(create_sym_state_constraint(suggestion_intermediate[0], sym_ltd, neg_sym_ltd))
            neg_pre_ltd = create_neg_sym_state_constraint(suggestion_intermediate[0], sym_ltd, neg_sym_ltd)

            all_posts_list = []
            for post in suggestion_intermediate[1]:
                all_posts_list.append(ltd.Next(create_sym_state_constraint(post, sym_ltd, neg_sym_ltd)))

            implication_next_ltd = ltd.Always(ltd.Or([neg_pre_ltd] + all_posts_list + [pre_ltd]), rollout_traj.shape[1]-1)
            implication_next_list.extend([implication_next_ltd])

        # Always stay in states in the specification
        always_list = []
        for state in self.suggestion_unique:
            always_list.append(create_sym_state_constraint(state, sym_ltd, neg_sym_ltd))
        if len(always_list) == 1:
            always_ltd = ltd.Always(always_list[0], rollout_traj.shape[1])
        else:
            always_ltd = ltd.Always(ltd.Or(always_list), rollout_traj.shape[1])

        final_list = []
        final_list.extend(implication_next_list)
        final_list.append(always_ltd)

        # Stretch LTD constraints:
        # The stretch dmps are in configuration/joint space instead of cartesian space.
        # We therefore need to constrain the end effector cartesian coordinate based on the joint angles
        stretch_limits_list = []
        if "stretch" in self.opts:
            final_list.append(ltd.Always(ltd.GEQ2(rollout_term, ltd.TermStatic(self.workspace_bnds[[2, 3, 4, 5], 0]), dim=np.array([2, 3, 4, 5])), rollout_traj.shape[1]))
            final_list.append(ltd.Always(ltd.LEQ2(rollout_term, ltd.TermStatic(self.workspace_bnds[[2, 3, 4, 5], 1]), dim=np.array([2, 3, 4, 5])), rollout_traj.shape[1]))

        if len(final_list) == 1:
            final_ltd = final_list[0]
        else:
            final_ltd = ltd.And(final_list)

        return final_ltd

    def domains(self, ins, targets):
        return fully_global_ins(ins, self.epsilon)


class AutomaticIntermediateSteps:
    def __init__(self, symbols, suggestion_intermediate, suggestion_unique, hard_constraints, workspace_bnds, epsilon):
        self.symbols = symbols
        self.suggestion_intermediate = suggestion_intermediate
        self.suggestion_unique = suggestion_unique
        self.hard_constraints = hard_constraints
        self.epsilon = epsilon
        self.workspace_bnds = workspace_bnds

    def condition(self, zs, ins, targets, net, rollout_func):
        weights = net(zs)
        rollout_traj = rollout_func(zs[:, 0], zs[:, -1], weights)[0]
        rollout_term = ltd.TermDynamic(rollout_traj)

        sym_ltd = dict()
        for sym_name, sym in self.symbols.items():
            bnds_list = []
            if sym.get_type() == 'rectangle':
                for dim in sym.get_dims():
                    bnds_list.append(ltd.GEQ2(rollout_term, ltd.TermStatic(sym.bounds[[dim], 0]), dim=np.array([dim])))
                    bnds_list.append(ltd.LEQ2(rollout_term, ltd.TermStatic(sym.bounds[[dim], 1]), dim=np.array([dim])))
            if sym.get_type() == 'circle':
                bnds_list.append(ltd.LT2(
                    ltd.TermDynamic(torch.norm(rollout_term.xs - sym.get_center(), dim=2, keepdim=True)),
                    ltd.TermStatic(sym.get_radius()), dim=np.array([0])))
            if sym.get_type() == "rectangle-ee":
                for dim in sym.get_dims():
                    l_wrist = rollout_term.xs[:, :, 3]
                    t_robot = rollout_term.xs[:, :, 2]
                    t_wrist = rollout_term.xs[:, :, 5]
                    x_robot = rollout_term.xs[:, :, 0]
                    y_robot = rollout_term.xs[:, :, 1]
                    l_ee = 0.1
                    # dim == 0 -> x
                    x_ee = l_ee * torch.cos(t_robot + t_wrist) + l_wrist * torch.sin(t_robot) + x_robot
                    y_ee = l_ee * torch.sin(t_robot + t_wrist) - l_wrist * torch.cos(t_robot) + y_robot
                    pos_ee = ltd.TermDynamic(torch.stack([x_ee, y_ee], dim=2))
                    bnds_list.append(ltd.GEQ2(pos_ee, ltd.TermStatic(sym.bounds[[dim], 0]), dim=np.array([dim])))
                    bnds_list.append(ltd.LEQ2(pos_ee, ltd.TermStatic(sym.bounds[[dim], 1]), dim=np.array([dim])))
            if sym.get_type() == "circle-ee":
                l_wrist = rollout_term.xs[:, :, 3]
                t_robot = rollout_term.xs[:, :, 2]
                t_wrist = rollout_term.xs[:, :, 5]
                x_robot = rollout_term.xs[:, :, 0]
                y_robot = rollout_term.xs[:, :, 1]
                l_ee = 0.1
                # dim == 0 -> x
                x_ee = l_ee * torch.cos(t_robot + t_wrist) + l_wrist * torch.sin(t_robot) + x_robot
                y_ee = l_ee * torch.sin(t_robot + t_wrist) - l_wrist * torch.cos(t_robot) + y_robot
                pos_ee = torch.stack([x_ee, y_ee], dim=2)
                bnds_list.append(ltd.LT2(
                    ltd.TermDynamic(torch.norm(pos_ee - sym.get_center(), dim=2, keepdim=True)),
                    ltd.TermStatic(sym.get_radius()), dim=np.array([0])))
            sym_ltd[sym_name] = ltd.And(bnds_list)
        neg_sym_ltd = dict()
        for sym_name, sym in self.symbols.items():
            bnds_list = []
            if sym.get_type() == 'rectangle':
                for dim in sym.get_dims():
                    multiplier = 1
                    bnds_list.append(ltd.LT2(rollout_term, ltd.TermStatic(sym.bounds[[dim], 0]), dim=np.array([dim])))
                    bnds_list.append(ltd.GT2(rollout_term, ltd.TermStatic(sym.bounds[[dim], 1]), dim=np.array([dim])))
            if sym.get_type() == 'circle':
                bnds_list.append(ltd.GEQ2(
                    ltd.TermDynamic(torch.norm(rollout_term.xs - sym.get_center(), dim=2, keepdim=True)),
                    ltd.TermStatic(sym.get_radius()), dim=np.array([0])))
            if sym.get_type() == "rectangle-ee":
                for dim in sym.get_dims():
                    l_wrist = rollout_term.xs[:, :, 3]
                    t_robot = rollout_term.xs[:, :, 2]
                    t_wrist = rollout_term.xs[:, :, 5]
                    x_robot = rollout_term.xs[:, :, 0]
                    y_robot = rollout_term.xs[:, :, 1]
                    l_ee = 0.1
                    # dim == 0 -> x
                    x_ee = l_ee * torch.cos(t_robot + t_wrist) + l_wrist * torch.sin(t_robot) + x_robot
                    y_ee = l_ee * torch.sin(t_robot + t_wrist) - l_wrist * torch.cos(t_robot) + y_robot
                    pos_ee = ltd.TermDynamic(torch.stack([x_ee, y_ee], dim=2))
                    bnds_list.append(ltd.LT2(pos_ee, ltd.TermStatic(sym.bounds[[dim], 0]), dim=np.array([dim])))
                    bnds_list.append(ltd.GT2(pos_ee, ltd.TermStatic(sym.bounds[[dim], 1]), dim=np.array([dim])))
            if sym.get_type() == "circle-ee":
                l_wrist = rollout_term.xs[:, :, 3]
                t_robot = rollout_term.xs[:, :, 2]
                t_wrist = rollout_term.xs[:, :, 5]
                x_robot = rollout_term.xs[:, :, 0]
                y_robot = rollout_term.xs[:, :, 1]
                l_ee = 0.1
                # dim == 0 -> x
                x_ee = l_ee * torch.cos(t_robot + t_wrist) + l_wrist * torch.sin(t_robot) + x_robot
                y_ee = l_ee * torch.sin(t_robot + t_wrist) - l_wrist * torch.cos(t_robot) + y_robot
                pos_ee = torch.stack([x_ee, y_ee], dim=2)
                bnds_list.append(ltd.GEQ2(
                    ltd.TermDynamic(torch.norm(pos_ee - sym.get_center(), dim=2, keepdim=True)),
                    ltd.TermStatic(sym.get_radius()), dim=np.array([0])))
            neg_sym_ltd[sym_name] = ltd.Or(bnds_list)

        # A post will come after the pres
        pre_list = []
        for p, val in self.suggestion_intermediate[0].items():
            if len(sym_ltd[p]) == 0:
                continue
            if val:
                pre_list.append(sym_ltd[p])
            else:
                pre_list.append(neg_sym_ltd[p])
        pre_ltd = ltd.And(pre_list)

        # A post will come after the pres
        neg_pre_list = []
        for p, val in self.suggestion_intermediate[0].items():
            if len(sym_ltd[p]) == 0:
                continue
            if val:
                neg_pre_list.append(neg_sym_ltd[p])
            else:
                neg_pre_list.append(sym_ltd[p])
        neg_pre_ltd = ltd.Or(neg_pre_list)

        all_posts_list = []
        for post in self.suggestion_intermediate[1]:
            post_list = []
            for p, val in post.items():
                if len(sym_ltd[p]) == 0:
                    continue
                if val:
                    post_list.append(sym_ltd[p])
                else:
                    post_list.append(neg_sym_ltd[p])
            all_posts_list.append(ltd.And(post_list))
        if len(all_posts_list) == 1:
            all_posts_ltd = all_posts_list[0]
        else:
            all_posts_ltd = ltd.Or(all_posts_list)

        until_ltd = ltd.Until1(pre_ltd, all_posts_ltd, rollout_traj.shape[1])
        implication_ltd = ltd.Always(ltd.Or([neg_pre_ltd, until_ltd]), rollout_traj.shape[1])
        # eventually_post_ltd = ltd.Eventually(all_posts_ltd, rollout_traj.shape[1])
        # eventually_pre_ltd = ltd.Eventually(pre_ltd, rollout_traj.shape[1])
        # final_ltd = ltd.And([implication_ltd, eventually_post_ltd, eventually_pre_ltd])

        return implication_ltd

    def domains(self, ins, targets):
        return fully_global_ins(ins, self.epsilon)

    def string(self):
        # Prints the specification in the format:
        # (!pre_sym1 | !pre_sym2 | ... ) -> (pre_sym1 & pre_sym2 & ...) U (post_cond1 | post_cond2 | ...)
        neg_pre_print = []
        pre_print = []
        for p, val in self.suggestion_intermediate[0].items():
            if val:
                neg_pre_print.append("!" + p)
                pre_print.append(p)
            else:
                neg_pre_print.append(p)
                pre_print.append("!" + p)

        post_print = []
        for post in self.suggestion_intermediate[1]:
            one_post_list = []
            for p, val in post.items():
                if val:
                    one_post_list.append(p)
                else:
                    one_post_list.append("!" + p)
            post_print.append("(" + " & ".join(one_post_list) + ")")
        return "({}) -> (({}) U ({}))".format(" | ".join(neg_pre_print), " & ".join(pre_print), " | ".join(post_print))


class States:
    def __init__(self, symbols, suggestions_pre, epsilon, buffer=0.2):
        self.symbols = symbols
        self.suggestions_pre = suggestions_pre
        self.epsilon = epsilon
        self.buffer = buffer

    def condition(self, point):
        sym_ltd = dict()
        for sym_name, sym in self.symbols.items():
            bnds_dim_list = []
            if sym.get_type() == 'rectangle':
                for dim in sym.get_dims():
                    bnds_dim_list.append(ltd.And([
                        ltd.GEQ2(point, ltd.TermStatic((torch.from_numpy(sym.bounds[:, 0]) + self.buffer)[[dim]]),
                                      dim=np.array([dim])),
                        ltd.LEQ2(point, ltd.TermStatic((torch.from_numpy(sym.bounds[:, 1]) - self.buffer)[[dim]]),
                                      dim=np.array([dim]))]))
            if sym.get_type() == 'circle':
                bnds_dim_list.append(
                    ltd.LT2(
                        ltd.TermStatic(torch.norm(point.x - torch.from_numpy(sym.get_center()), dim=1, keepdim=True)),
                        ltd.TermStatic(torch.from_numpy(sym.get_radius() - self.buffer)), dim=[0])
                )
            if sym.get_type() == "rectangle-ee":
                for dim in sym.get_dims():
                    l_wrist = point.x[0, 3]
                    t_robot = point.x[0, 2]
                    t_wrist = point.x[0, 5]
                    x_robot = point.x[0, 0]
                    y_robot = point.x[0, 1]
                    l_ee = 0.1
                    # dim == 0 -> x
                    x_ee = l_ee * torch.cos(t_robot + t_wrist) + l_wrist * torch.sin(t_robot) + x_robot
                    y_ee = l_ee * torch.sin(t_robot + t_wrist) - l_wrist * torch.cos(t_robot) + y_robot
                    pos_ee = ltd.TermStatic(torch.stack([x_ee, y_ee])[None, :])
                    bnds_dim_list.append(ltd.GEQ2(pos_ee, ltd.TermStatic(torch.from_numpy(sym.bounds[[dim], 0] + self.buffer)), dim=np.array([dim])))
                    bnds_dim_list.append(ltd.LEQ2(pos_ee, ltd.TermStatic(torch.from_numpy(sym.bounds[[dim], 1] - self.buffer)), dim=np.array([dim])))
            if sym.get_type() == "circle-ee":
                l_wrist = point.x[0, 3]
                t_robot = point.x[0, 2]
                t_wrist = point.x[0, 5]
                x_robot = point.x[0, 0]
                y_robot = point.x[0, 1]
                l_ee = 0.1
                # dim == 0 -> x
                x_ee = l_ee * torch.cos(t_robot + t_wrist) + l_wrist * torch.sin(t_robot) + x_robot
                y_ee = l_ee * torch.sin(t_robot + t_wrist) - l_wrist * torch.cos(t_robot) + y_robot
                pos_ee = torch.stack([x_ee, y_ee])
                bnds_dim_list.append(ltd.LT2(
                    ltd.TermStatic(torch.norm(pos_ee - torch.from_numpy(sym.get_center()), dim=2, keepdim=True)),
                    ltd.TermStatic(torch.from_numpy(sym.get_radius() - self.buffer)), dim=np.array([0])))
            sym_ltd[sym_name] = ltd.And(bnds_dim_list)
            # sym_ltd[sym_name] = ltd.And([ltd.GEQ2(point, ltd.TermStatic(torch.from_numpy(sym['bnds'][:, 0]) + self.buffer * (torch.from_numpy(sym['bnds'][:, 1]) - torch.from_numpy(sym['bnds'][:, 0]))), dim=sym['dims']),
            #                              ltd.LEQ2(point, ltd.TermStatic(torch.from_numpy(sym['bnds'][:, 1]) - self.buffer * (torch.from_numpy(sym['bnds'][:, 1]) - torch.from_numpy(sym['bnds'][:, 0]))), dim=sym['dims'])])

        all_pres_list = []
        for pre in self.suggestions_pre:
            pre_list = []
            for p, val in pre.items():
                if len(sym_ltd[p]) == 0:
                    continue
                if val:
                    pre_list.append(sym_ltd[p])
                else:
                    pre_list.append(ltd.Negate(sym_ltd[p]))
            all_pres_list.append(ltd.And(pre_list))
        return ltd.Or(all_pres_list)

    def domains(self, ins, targets):
        return fully_global_ins(ins, self.epsilon)


class AlwaysFormula:
    """
    This will create the constraints on a trajectory to obey the skill.

    Constraint:
    Formula is dict of true/false, everything else is not constrained

    """
    def __init__(self, symbols, formula, epsilon):
        self.symbols = symbols
        self.formula = formula
        self.epsilon = epsilon

    def condition(self, zs, ins, targets, net, rollout_func):
        weights = net(zs)

        # The actual trajectory
        rollout_traj = rollout_func(zs[:, 0], zs[:, -1], weights)[0]
        rollout_term = ltd.TermDynamic(rollout_traj)

        # Each symbol corresponds to a constraint
        sym_ltd, neg_sym_ltd = create_sym_and_neg_ltd_dicts(self.symbols, rollout_term)

        ltd_formula = create_sym_state_constraint(self.formula, sym_ltd, neg_sym_ltd)

        return ltd.Always(ltd_formula, rollout_traj.shape[1])

    def domains(self, ins, targets):
        return fully_global_ins(ins, self.epsilon)

class EventuallyOrFormulas:
    """
    This will create the constraints on a trajectory to obey the skill.

    Constraint:
    Formula is dict of true/false, everything else is not constrained

    """
    def __init__(self, symbols, formulas, epsilon):
        self.symbols = symbols
        self.formulas = formulas
        self.epsilon = epsilon

    def condition(self, zs, ins, targets, net, rollout_func):
        weights = net(zs)

        # The actual trajectory
        rollout_traj = rollout_func(zs[:, 0], zs[:, -1], weights)[0]
        rollout_term = ltd.TermDynamic(rollout_traj)

        # Each symbol corresponds to a constraint
        sym_ltd, neg_sym_ltd = create_sym_and_neg_ltd_dicts(self.symbols, rollout_term)

        ltd_formulas = []
        for formula in self.formulas:
            ltd_formulas.append(create_sym_state_constraint(formula, sym_ltd, neg_sym_ltd))

        if len(ltd_formulas) == 1:
            ltd_formulas_or = ltd_formulas[0]
        else:
            ltd_formulas_or = ltd.Or(ltd_formulas)

        return ltd.Eventually(ltd_formulas_or, rollout_traj.shape[1])

    def domains(self, ins, targets):
        return fully_global_ins(ins, self.epsilon)

class AndEventuallyFormulas:
    """
    This will create the constraints on a trajectory to obey the skill.

    Constraint:
    Formula is dict of true/false, everything else is not constrained

    """
    def __init__(self, symbols, formulas, epsilon):
        self.symbols = symbols
        self.formulas = formulas
        self.epsilon = epsilon

    def condition(self, zs, ins, targets, net, rollout_func):
        weights = net(zs)

        # The actual trajectory
        rollout_traj = rollout_func(zs[:, 0], zs[:, -1], weights)[0]
        rollout_term = ltd.TermDynamic(rollout_traj)

        # Each symbol corresponds to a constraint
        sym_ltd, neg_sym_ltd = create_sym_and_neg_ltd_dicts(self.symbols, rollout_term)

        ltd_formulas = []
        for formula in self.formulas:
            ltd_formulas.append(ltd.Eventually(create_sym_state_constraint(formula, sym_ltd, neg_sym_ltd), rollout_traj.shape[1]))

        if len(ltd_formulas) == 1:
            ltd_formulas_and = ltd_formulas[0]
        else:
            ltd_formulas_and = ltd.And(ltd_formulas)

        return ltd_formulas_and

    def domains(self, ins, targets):
        return fully_global_ins(ins, self.epsilon)


class SequenceFormulas:
    """
    This will create the constraints on a trajectory to obey the skill.

    Constraint:
    Formula is dict of true/false, everything else is not constrained

    """
    def __init__(self, symbols, formulas, epsilon):
        self.symbols = symbols
        self.formulas = formulas
        self.epsilon = epsilon

    def condition(self, zs, ins, targets, net, rollout_func):
        weights = net(zs)

        # The actual trajectory
        rollout_traj = rollout_func(zs[:, 0], zs[:, -1], weights)[0]
        rollout_term = ltd.TermDynamic(rollout_traj)

        # Each symbol corresponds to a constraint
        sym_ltd, neg_sym_ltd = create_sym_and_neg_ltd_dicts(self.symbols, rollout_term)

        ltd_formulas = []
        ltd_formulas_neg = []
        for formula in self.formulas:
            ltd_formulas.append(create_sym_state_constraint(formula, sym_ltd, neg_sym_ltd))
            ltd_formulas_neg.append(create_neg_sym_state_constraint(formula, sym_ltd, neg_sym_ltd))

        # A postcondition or the same precondition will always come after a given precondition
        # pre -> N (pre | posts) is the same as !pre | N (pre | posts)
        sequence_list = []
        for ii in range(len(self.formulas) - 1):
            # A post will come after the pres

            one_step = ltd.Always(ltd.Or([ltd_formulas_neg[ii], ltd.Next(ltd.Or([ltd_formulas[ii], ltd_formulas[ii + 1]]))]), rollout_traj.shape[1] - 1)
            sequence_list.append(one_step)

        if len(sequence_list) == 1:
            ltd_formulas_and = sequence_list[0]
        else:
            ltd_formulas_and = ltd.And(sequence_list)

        return ltd_formulas_and

    def domains(self, ins, targets):
        return fully_global_ins(ins, self.epsilon)


class SequenceFormulasAndAlways:
    """
    This will create the constraints on a trajectory to obey the skill.

    Constraint:
    Formula is dict of true/false, everything else is not constrained

    """
    def __init__(self, symbols, formulas_sequence, formulas_always, epsilon):
        self.symbols = symbols
        self.formulas_sequence = formulas_sequence
        self.formulas_always = formulas_always
        self.epsilon = epsilon

    def condition(self, zs, ins, targets, net, rollout_func):
        weights = net(zs)

        # The actual trajectory
        rollout_traj = rollout_func(zs[:, 0], zs[:, -1], weights)[0]
        rollout_term = ltd.TermDynamic(rollout_traj)

        # Each symbol corresponds to a constraint
        sym_ltd, neg_sym_ltd = create_sym_and_neg_ltd_dicts(self.symbols, rollout_term)

        # The sequence part
        ltd_formulas = []
        ltd_formulas_neg = []
        for formula in self.formulas_sequence:
            ltd_formulas.append(create_sym_state_constraint(formula, sym_ltd, neg_sym_ltd))
            ltd_formulas_neg.append(create_neg_sym_state_constraint(formula, sym_ltd, neg_sym_ltd))

        # A postcondition or the same precondition will always come after a given precondition
        # pre -> N (pre | posts) is the same as !pre | N (pre | posts)
        sequence_list = []
        for ii in range(len(self.formulas_sequence) - 1):
            # A post will come after the pres

            one_step = ltd.Always(ltd.Or([ltd_formulas_neg[ii], ltd.Next(ltd.Or([ltd_formulas[ii], ltd_formulas[ii + 1]]))]), rollout_traj.shape[1] - 1)
            sequence_list.append(one_step)

        # The always part
        always_list = []
        for formula in self.formulas_always:
            always_list.append(create_sym_state_constraint(formula, sym_ltd, neg_sym_ltd))
        if len(always_list) == 1:
            ltd_always_or = always_list[0]
        else:
            ltd_always_or = ltd.Or(always_list)

        formula_list = []
        formula_list.extend(sequence_list)
        formula_list.append(ltd.Always(ltd_always_or, rollout_traj.shape[1]))
        if len(formula_list) == 1:
            ltd_formulas_and = formula_list[0]
        else:
            ltd_formulas_and = ltd.And(formula_list)

        return ltd_formulas_and

    def domains(self, ins, targets):
        return fully_global_ins(ins, self.epsilon)


class SequenceFormulasMultiplePostsAndAlways:
    """
    This will create the constraints on a trajectory to obey the skill.

    Constraint:
    Formula is dict of true/false, everything else is not constrained

    """
    def __init__(self, symbols, formulas_sequence, formulas_always, epsilon):
        self.symbols = symbols
        self.formulas_sequence = formulas_sequence
        self.formulas_always = formulas_always
        self.epsilon = epsilon

    def condition(self, zs, ins, targets, net, rollout_func):
        weights = net(zs)

        # The actual trajectory
        rollout_traj = rollout_func(zs[:, 0], zs[:, -1], weights)[0]
        rollout_term = ltd.TermDynamic(rollout_traj)

        # Each symbol corresponds to a constraint
        sym_ltd, neg_sym_ltd = create_sym_and_neg_ltd_dicts(self.symbols, rollout_term)

        # The sequence part
        # A postcondition or the same precondition will always come after a given precondition
        # pre -> N (pre | posts) is the same as !pre | N (pre | posts)
        sequence_list = []
        for pre, posts in self.formulas_sequence:
            # A post will come after the pres
            ltd_pos_pre = create_sym_state_constraint(pre, sym_ltd, neg_sym_ltd)
            ltd_neg_pre = create_neg_sym_state_constraint(pre, sym_ltd, neg_sym_ltd)
            post_list = []
            for post in posts:
                post_list.append(create_sym_state_constraint(post, sym_ltd, neg_sym_ltd))
            ltd_next_or_list = [ltd_pos_pre]
            ltd_next_or_list.extend(post_list)

            one_step = ltd.Always(ltd.Or([ltd_neg_pre, ltd.Next(ltd.Or(ltd_next_or_list))]), rollout_traj.shape[1] - 1)
            sequence_list.append(one_step)

        # The always part
        always_list = []
        ltd_always_or = None
        if self.formulas_always is not None:

            for formula in self.formulas_always:
                always_list.append(create_sym_state_constraint(formula, sym_ltd, neg_sym_ltd))
            if len(always_list) == 1:
                ltd_always_or = always_list[0]
            else:
                ltd_always_or = ltd.Or(always_list)

        formula_list = []
        formula_list.extend(sequence_list)
        if ltd_always_or is not None:
            formula_list.append(ltd.Always(ltd_always_or, rollout_traj.shape[1]))
        if len(formula_list) == 1:
            ltd_formulas_and = formula_list[0]
        else:
            ltd_formulas_and = ltd.And(formula_list)

        return ltd_formulas_and

    def domains(self, ins, targets):
        return fully_global_ins(ins, self.epsilon)


class SequenceFormulasMultiplePostsAndAlwaysWithIK:
    """
    This will create the constraints on a trajectory to obey the skill.

    Constraint:
    Formula is dict of true/false, everything else is not constrained

    """
    def __init__(self, symbols, formulas_sequence, formulas_always, epsilon):
        self.symbols = symbols
        self.formulas_sequence = formulas_sequence
        self.formulas_always = formulas_always
        self.epsilon = epsilon
        self.max_ext = 0.91

    def condition(self, zs, ins, targets, net, rollout_func):
        weights = net(zs)

        # The actual trajectory
        rollout_traj = rollout_func(zs[:, 0], zs[:, -1], weights)[0]
        rollout_term = ltd.TermDynamic(rollout_traj)

        # Each symbol corresponds to a constraint
        sym_ltd, neg_sym_ltd = create_sym_and_neg_ltd_dicts(self.symbols, rollout_term)

        # The sequence part
        # A postcondition or the same precondition will always come after a given precondition
        # pre -> N (pre | posts) is the same as !pre | N (pre | posts)
        sequence_list = []
        for pre, posts in self.formulas_sequence:
            # A post will come after the pres
            ltd_pos_pre = create_sym_state_constraint(pre, sym_ltd, neg_sym_ltd)
            ltd_neg_pre = create_neg_sym_state_constraint(pre, sym_ltd, neg_sym_ltd)
            post_list = []
            for post in posts:
                post_list.append(create_sym_state_constraint(post, sym_ltd, neg_sym_ltd))
            ltd_next_or_list = [ltd_pos_pre]
            ltd_next_or_list.extend(post_list)

            one_step = ltd.Always(ltd.Or([ltd_neg_pre, ltd.Next(ltd.Or(ltd_next_or_list))]), rollout_traj.shape[1] - 1)
            sequence_list.append(one_step)

        # The always part
        always_list = []
        for formula in self.formulas_always:
            always_list.append(create_sym_state_constraint(formula, sym_ltd, neg_sym_ltd))
        if len(always_list) == 1:
            ltd_always_or = always_list[0]
        else:
            ltd_always_or = ltd.Or(always_list)

        # ltd_Ik
        est_ext_squared = ltd.TermDynamic(torch.square(rollout_term.xs[:, :, 0:1] - rollout_term.xs[:, :, 2:3]) + torch.square(rollout_term.xs[:, :, 1:2] - rollout_term.xs[:, :, 3:4]))
        ltd_ik = ltd.Always(ltd.LT2(est_ext_squared, ltd.TermStatic(torch.Tensor([self.max_ext * self.max_ext])), dim=[0]), rollout_traj.shape[1])


        formula_list = []
        formula_list.extend(sequence_list)
        formula_list.append(ltd.Always(ltd_always_or, rollout_traj.shape[1]))
        formula_list.append(ltd_ik)
        if len(formula_list) == 1:
            ltd_formulas_and = formula_list[0]
        else:
            ltd_formulas_and = ltd.And(formula_list)

        return ltd_formulas_and

    def domains(self, ins, targets):
        return fully_global_ins(ins, self.epsilon)


# def create_sym_constraint(sym, rollout_term):
#     """
#     Creates the list of constraints a symbol has
#
#     #TODO: Circles
#
#     :return:
#     """
#     bnds_list = []
#     if sym.get_type() == 'rectangle':
#         if sym.transform is None:
#             if len(sym.get_dims()) == 2:
#                 rollout_term_selected = ltd.TermDynamic(rollout_term.xs[:, :, sym.get_dims()])
#                 return ltd.InRectangle(rollout_term_selected, ltd.TermStatic([sym.bounds[0, 0], sym.bounds[1, 0], sym.bounds[0, 1], sym.bounds[1, 1]]))
#             for ii, dim in enumerate(sym.get_dims()):
#                 bnds_list.append(ltd.GEQ2(rollout_term, ltd.TermStatic(sym.bounds[[ii], 0]), dim=np.array([dim])))
#                 bnds_list.append(ltd.LEQ2(rollout_term, ltd.TermStatic(sym.bounds[[ii], 1]), dim=np.array([dim])))
#         elif sym.transform in ['ee']:
#             # Transform via stretch fk
#             ee_traj = ltd.TermDynamic(fk_stretch(rollout_term.xs))
#             if len(sym.get_plot_dims()) == 2:
#                 rollout_term_selected = ltd.TermDynamic(ee_traj.xs[:, :, sym.get_plot_dims()])
#                 return ltd.InRectangle(rollout_term_selected, ltd.TermStatic([sym.bounds[0, 0], sym.bounds[1, 0], sym.bounds[0, 1], sym.bounds[1, 1]]))
#             for ii, dim in enumerate(sym.get_plot_dims()):
#                 bnds_list.append(ltd.GEQ2(ee_traj, ltd.TermStatic(sym.bounds[[ii], 0]), dim=np.array([ii])))
#                 bnds_list.append(ltd.LEQ2(ee_traj, ltd.TermStatic(sym.bounds[[ii], 1]), dim=np.array([ii])))
#
#     return ltd.And(bnds_list)


def create_sym_constraint(sym, rollout_term):
    """
    Creates the list of constraints a symbol has

    #TODO: Circles

    :return:
    """
    bnds_list = []
    if sym.get_type() == 'rectangle':
        if sym.transform is None:
            for ii, dim in enumerate(sym.get_dims()):
                bnds_list.append(ltd.GEQ2(rollout_term, ltd.TermStatic(sym.bounds[[ii], 0]), dim=np.array([dim])))
                bnds_list.append(ltd.LEQ2(rollout_term, ltd.TermStatic(sym.bounds[[ii], 1]), dim=np.array([dim])))
        elif sym.transform in ['ee']:
            # Transform via stretch fk
            ee_traj = ltd.TermDynamic(fk_stretch(rollout_term.xs))
            for ii, dim in enumerate(sym.get_plot_dims()):
                bnds_list.append(ltd.GEQ2(ee_traj, ltd.TermStatic(sym.bounds[[ii], 0]), dim=np.array([ii])))
                bnds_list.append(ltd.LEQ2(ee_traj, ltd.TermStatic(sym.bounds[[ii], 1]), dim=np.array([ii])))

    return ltd.And(bnds_list)


def create_neg_sym_constraint(sym, rollout_term):
    """
    Creates the list of constraints a symbol has

    #TODO: Circles

    :return:
    """
    bnds_list = []
    if sym.get_type() == 'rectangle':
        if sym.transform is None:
            for ii, dim in enumerate(sym.get_dims()):
                bnds_list.append(ltd.LT2(rollout_term, ltd.TermStatic(sym.bounds[[ii], 0]), dim=np.array([dim])))
                bnds_list.append(ltd.GT2(rollout_term, ltd.TermStatic(sym.bounds[[ii], 1]), dim=np.array([dim])))
        elif sym.transform in ['ee']:
            # Transform via stretch fk
            ee_traj = ltd.TermDynamic(fk_stretch(rollout_term.xs))
            for ii, dim in enumerate(sym.get_plot_dims()):
                bnds_list.append(ltd.LT2(ee_traj, ltd.TermStatic(sym.bounds[[ii], 0]), dim=np.array([ii])))
                bnds_list.append(ltd.GT2(ee_traj, ltd.TermStatic(sym.bounds[[ii], 1]), dim=np.array([ii])))

    return ltd.Or(bnds_list)

def create_sym_state_constraint(sym_state, sym_ltd, neg_sym_ltd):
    constraint_list = []
    for sym, truth_value in sym_state.items():
        # if len(sym_ltd[sym].exprs) == 0:
        #     continue
        if truth_value:
            constraint_list.extend([sym_ltd[sym]])
        else:
            constraint_list.extend([neg_sym_ltd[sym]])

    return ltd.And(constraint_list)

def create_neg_sym_state_constraint(sym_state, sym_ltd, neg_sym_ltd):
    constraint_list = []
    for sym, truth_value in sym_state.items():
        # if len(sym_ltd[sym].exprs) == 0:
        #     continue
        if truth_value:
            constraint_list.extend([neg_sym_ltd[sym]])
        else:
            constraint_list.extend([sym_ltd[sym]])

    return ltd.Or(constraint_list)

def fk_stretch(rollout):
    """
    Foward kinematics of the stretch

    :param rollout:
    :return:
    """
    wrist_x_in_base_frame = 0.14
    wrist_y_in_base_frame = -0.16
    gripper_length_xy = 0.23
    gripper_offset_z = -0.1
    base_height_z = 0.05

    x_robot = rollout[:, :, 0]
    y_robot = rollout[:, :, 1]
    t_robot = rollout[:, :, 2]
    arm_extension = rollout[:, :, 3]
    lift = rollout[:, :, 4]
    t_wrist = rollout[:, :, 5]
    x_ee_in_robot = wrist_x_in_base_frame + gripper_length_xy * torch.sin(t_wrist)
    y_ee_in_robot = wrist_y_in_base_frame - arm_extension - gripper_length_xy * torch.cos(t_wrist)

    x_ee_in_global = x_ee_in_robot * torch.cos(t_robot) - y_ee_in_robot * torch.sin(t_robot) + x_robot
    y_ee_in_global = x_ee_in_robot * torch.sin(t_robot) + y_ee_in_robot * torch.cos(t_robot) + y_robot
    z_ee_in_global = lift + gripper_offset_z + base_height_z

    return torch.stack([x_ee_in_global, y_ee_in_global, z_ee_in_global], dim=2)

def create_sym_and_neg_ltd_dicts(symbols, rollout_term):
    sym_ltd = dict()
    for sym_name, sym in symbols.items():
        one_sym_ltd = create_sym_constraint(sym, rollout_term)
        sym_ltd[sym_name] = one_sym_ltd

    neg_sym_ltd = dict()
    for sym_name, sym in symbols.items():
        one_sym_neg_ltd = create_neg_sym_constraint(sym, rollout_term)
        neg_sym_ltd[sym_name] = one_sym_neg_ltd

    return sym_ltd, neg_sym_ltd

