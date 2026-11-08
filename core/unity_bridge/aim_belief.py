"""Candidate estimate from perceived target and ideal plan, without truth/RNG."""
import numpy as np
from src.agent.module_aim import Aim
from src.configs.constants import AGENT_STATE_ALIAS as AS, TINTERVAL


def displacement_at(positions,velocities,when):
    """Sample the same interpolation at one point without expanding the whole plan."""
    if len(positions)<2:return positions[0]-positions[0]
    stride=TINTERVAL.MUSCLE//TINTERVAL.INTERP1
    index=min(max(int(np.ceil(when))-1,0),(len(positions)-1)*stride)
    # At a segment boundary the old concatenate takes the preceding segment's
    # endpoint. Keep that exact choice, including its floating-point rounding.
    segment=max(index-1,0)//stride
    local=index-segment*stride
    p,_=Aim.interp_segment(positions[segment],velocities[segment],
        positions[segment+1],velocities[segment+1],TINTERVAL.MUSCLE,TINTERVAL.INTERP1)
    return p[local]-positions[0]


def estimate_candidate(agent,state,delay_ms,weapon,hp,critical_radius):
    if delay_ms<=TINTERVAL.BUMP:
        hand=displacement_at(agent.ongoing_motor_planned_movement_pos,agent.ongoing_motor_planned_movement_vel,delay_ms)
    else:
        ideal_pos=np.concatenate((state[AS.HTRAJ_IDEAL_P],agent.waiting_motor_planned_movement_pos[1:]))
        ideal_vel=np.concatenate((state[AS.HTRAJ_IDEAL_V],agent.waiting_motor_planned_movement_vel[1:]))
        hand=(agent.ongoing_motor_planned_movement_pos[-1]-agent.ongoing_motor_planned_movement_pos[0]
              +displacement_at(ideal_pos,ideal_vel,delay_ms-TINTERVAL.BUMP))
    target_hat=np.asarray(state[agent._STATE_TARGET_POS_0_HAT],dtype=float)
    target_future=target_hat+np.asarray(state[AS.TARGET_VEL_HAT_BY_ORBIT_MM_S])*delay_ms/1000.
    projected=agent.game_env.target_monitor_position(initial_target_pos_monitor_mm=target_future,
        hand_displacement_mm=hand,target_orbit_angle_deg=0,clip_ratio=2.)
    crosshair_hat=np.asarray(state['crosshair_pos_hat_mm'],dtype=float)
    ratio=float(np.linalg.norm(projected-crosshair_hat))/max(agent.game_env.target_radius_mm,1e-6)
    critical=bool(ratio<=critical_radius)
    damage=(hp if critical and weapon['instantKill'] else
            weapon['minimum']+(weapon['maximum']-weapon['minimum'])*(1-ratio)**2) if ratio<=1 else 0.
    return dict(normalized_error=ratio,damage=float(damage),critical=critical,
                source='aim_perception_and_ideal_plan')
