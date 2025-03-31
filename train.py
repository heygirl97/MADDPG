import argparse
import xlwt
import xlrd
import matplotlib.pyplot as plt
import os
import math
import numpy as np
import tensorflow.compat.v1 as tf
from mpl_toolkits.mplot3d import Axes3D
tf.disable_eager_execution()         #可以用于从TensorFlow 1.x到2.x的复杂迁移项目的程序开头
import time
import pickle
import tf_slim
import maddpg.common.tf_util as U
from maddpg.trainer.maddpg import MADDPGAgentTrainer
#import tensorflow.contrib.layers as layers
import tf_slim as layers
import time

def parse_args():
    parser = argparse.ArgumentParser("Reinforcement Learning experiments for multiagent environments")
    # Environment
    parser.add_argument("--scenario", type=str, default="GX3vsGS3-2", help="name of the scenario script")
    parser.add_argument("--max-episode-len", type=int, default=30, help="maximum episode length")
    parser.add_argument("--num-episodes", type=int, default=2000000, help="number of episodes")
    parser.add_argument("--num-adversaries", type=int, default=1, help="number of adversaries")#敌人数量
    parser.add_argument("--good-policy", type=str, default="maddpg", help="policy for good agents")
    parser.add_argument("--adv-policy", type=str, default="maddpg", help="policy of adversaries")
    # Core training parameters
    parser.add_argument("--lr", type=float, default=1e-3, help="learning rate for Adam optimizer")#Adma 优化器学习效率
    parser.add_argument("--gamma", type=float, default=0.95, help="discount factor") #折扣系数
    parser.add_argument("--batch-size", type=int, default=2048, help="number of episodes to optimize at the same time")#一次训练所选取的样本数
    parser.add_argument("--num-units", type=int, default=128, help="number of units in the mlp")
    # Checkpointing
    parser.add_argument("--exp-name", type=str, default='GX3vsGS3-2', help="name of the experiment")#实验名称
    parser.add_argument("--save-dir", type=str, default="/home/swarm/Algorithm/ET_MADDPG_AD(GX2vsGS2)/tmpET2-DIvsBA/policy/", help="directory in which training state and model should be saved")#保存训练状态和神经网络模型的目录
    parser.add_argument("--save-rate", type=int, default=1000, help="save model once every time this many episodes are completed")#每过1000次保存一次网络模型数据
    parser.add_argument("--load-dir", type=str, default="", help="directory in which training state and model are loaded")#加载训练状态和模型的目录
    # Evaluation
    parser.add_argument("--restore", action="store_true", default=True)  #True  False
    parser.add_argument("--benchmark", action="store_true", default=False)
    parser.add_argument("--benchmark-iters", type=int, default=100000, help="number of i terations run for benchmarking")
    parser.add_argument("--benchmark-dir", type=str, default="./benchmark_files/", help="directory where benchmark data is saved")
    parser.add_argument("--plots-dir", type=str, default="./learning_curves/", help="directory where plot data is saved")#画图数据保存目录
    parser.add_argument("--savestate", action="store_true", default=True)
    parser.add_argument("--plotgj", action="store_true", default=True)  #True  False
    parser.add_argument("--test", action="store_true", default=False)
    parser.add_argument("--save-number", type=int, default=10, help="写入excel存储的轨迹数量")
    return parser.parse_args()

def plotguiji(Date_Number,date_number,savetimes,file_name):
    xlsx = xlrd.open_workbook(file_name)
    table = xlsx.sheet_by_index(0)
    print(date_number)
    # 获取agent1的dv 
    agent1px=[]
    agent1py=[]
    # plt.figure(1, figsize=(32, 20))#画第一个图
    for i in range(0,savetimes):
        agent1ax=[]
        agent1ay=[]
        agent1az=[]
        agent1ax0=[]
        agent1ay0=[]     
        agent1az0=[]   
        agent1_0 = [(table.cell_value(Date_Number[i]+1, m)) for m in range(0, 3)]
        agent1ax0.append(agent1_0[0])
        agent1ay0.append(agent1_0[1])
        agent1az0.append(agent1_0[2])
        agent2dx=[]
        agent2dy=[]
        agent2dz=[]
        agent2dx0=[]
        agent2dy0=[]        
        agent2dz0=[]    
        agent2_0 = [(table.cell_value(Date_Number[i]+1, m)) for m in range(3, 6)]
        agent2dx0.append(agent2_0[0])
        agent2dy0.append(agent2_0[1])
        agent2dz0.append(agent2_0[2])
        
        for j in range(Date_Number[i]+1,Date_Number[i+1]+1):
            agent1 = [(table.cell_value(j, m)) for m in range(0, 3)]
            agent1ax.append(agent1[0])
            agent1ay.append(agent1[1])
            agent1az.append(agent1[2])
        for l in range(Date_Number[i]+1,Date_Number[i+1]+1):
            agent2 = [(table.cell_value(l, m)) for m in range(3, 6)]
            agent2dx.append(agent2[0])
            agent2dy.append(agent2[1])
            agent2dz.append(agent2[2])
        
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(agent1ax, agent1ay, agent1az, c='r', marker='o')
        ax.plot(agent2dx, agent2dy, agent2dz, c='b', marker='o')
        ax.scatter(agent1ax[0], agent1ay[0], agent1az[0], c='r',s=200 ,marker='*')
        ax.scatter(agent2dx[0], agent2dy[0], agent2dz[0], c='b',s=200, marker='*')
        ax.scatter(agent1ax[-1], agent1ay[-1], agent1az[-1], c='r',s=100 ,marker='^')
        ax.scatter(agent2dx[-1], agent2dy[-1], agent2dz[-1], c='b',s=100, marker='^')
        # 设置半径
        radius = 1000
        # 指定起始和结束值，生成等间隔的经纬度序列
        longitude = np.linspace(0, 2 * np.pi, 200)
        latitude = np.linspace(0, np.pi, 200)

        # 计算向量间的外积
        x = radius * np.outer(np.cos(longitude), np.sin(latitude))
        y = radius * np.outer(np.sin(longitude), np.sin(latitude))
        z = radius * np.outer(np.ones(np.size(longitude)), np.cos(latitude))
        # 绘制球体
        ax.plot_surface(x, y, z, color='b', rstride=4, cstride=4, alpha=0.5)
        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Z Label')
        ax.set_aspect("equal")
        plt.show()


    #     # Plot the actual value
    #     plt.plot(agent1ax,agent1ay, 'b-',linewidth =2.0,markersize=8)
    #     plt.plot(agent2dx,agent2dy, 'r-',linewidth =2.0,markersize=8)
    #     plt.plot(agent3px,agent3py, 'k-',linewidth =2.0,markersize=8)
    #     plt.plot(agent1px0,agent1py0, 'b*',markersize=20)
    #     plt.plot(agent2px0,agent2py0, 'r*',markersize=20)
    #     plt.plot(agent3px0,agent3py0, 'k*',markersize=20)
    # plt.tick_params(axis='both', which='both',labelsize=30, length=5, width=1, direction='in')
    # label=['P1','P2','E']
    # plt.legend(label, loc = 'best', prop = {'size':40}) 
    # # 指定图片保存路径
    # figure_save_path = "file_fig"
    # if not os.path.exists(figure_save_path):
    #     os.makedirs(figure_save_path) # 如果不存在目录figure_save_path，则创建
    # plt.savefig(os.path.join(figure_save_path , '135-500(xy).png'))#第一个是指存储路径，第二个是图片名字
    # plt.show
    # return

def mlp_model(input, num_outputs, scope, reuse=False, num_units=128, rnn_cell=None):
    # This model takes as input an observation and returns values of all actions
    with tf.variable_scope(scope, reuse=reuse):
        out = input
        out = layers.fully_connected(out, num_outputs=num_units, activation_fn=tf.nn.relu)
        out = layers.fully_connected(out, num_outputs=num_units, activation_fn=tf.nn.relu)
        out = layers.fully_connected(out, num_outputs=num_outputs, activation_fn=None)
        return out

def make_env(scenario_name, arglist, benchmark=False):
    from multiagent.environmentET import MultiAgentEnv
    import multiagent.scenarios as scenarios

    # load scenario from script
    scenario = scenarios.load(scenario_name + ".py").Scenario()  #环境场景加载，（进入14行中参数定义的场景文件中）
    # create world
    world = scenario.make_world()
    # create multiagent environment
    if benchmark:
        env = MultiAgentEnv(world, scenario.fuel_n, scenario.reset_world, scenario.reward, scenario.observation, scenario.benchmark_data)
    else:
        env = MultiAgentEnv(world, scenario.fuel_n, scenario.reset_world, scenario.reward, scenario.observation, scenario.done_n, scenario.success_n, scenario.Fail, scenario.success_attacker)
    return env

def get_trainers(env, num_adversaries, obs_shape_n, arglist):
    trainers = []
    model = mlp_model
    trainer = MADDPGAgentTrainer
#   =============================================================================
#   给追击者配置训练网络
    for i in range(num_adversaries):
        print("agent_%d" % i)
        trainers.append(trainer(
            "agent_%d" % i, model, obs_shape_n, env.action_space, i, arglist,
            local_q_func=(arglist.adv_policy=='maddpg')))
#   =============================================================================

#   =============================================================================
#   给逃跑者配置训练网络
    for i in range(num_adversaries, env.n):
        trainers.append(trainer(
            "agent_%d" % i, model, obs_shape_n, env.action_space, i, arglist,
            local_q_func=(arglist.good_policy=='maddpg')))
    return trainers
#   =============================================================================


def train(arglist):
    # with U.make_session(8):  #分配计算CPU
    with U.make_session_gpu():
        # Create environment
        env = make_env(arglist.scenario, arglist, arglist.benchmark)
        # Create agent trainers
        obs_shape_n = [env.observation_space[i].shape for i in range(env.n)]
        num_adversaries = min(env.n, arglist.num_adversaries)
        trainers = get_trainers(env, num_adversaries, obs_shape_n, arglist)
        print('Using good policy {} and adv policy {}'.format(arglist.good_policy, arglist.adv_policy))

        # Initialize
        U.initialize()    #在全局作用域中初始化所有未初始化的变量

        # Load previous results, if necessary
        if arglist.load_dir == "":#/tmp1/policy/
            arglist.load_dir = arglist.save_dir
        if arglist.test or arglist.savestate or arglist.restore:
            print('Loading previous state...')
            U.load_state(arglist.load_dir)

        episode_rewards = [0.0]  # sum of rewards for all agents
        agent_rewards = [[0.0] for _ in range(env.n)]  # individual agent reward
        final_ep_rewards = []  # sum of rewards for training curve
        final_ep_ag_rewards = []  # agent rewards for training curve
        agent_info = [[[]]]  # placeholder for benchmarking info
        saver = tf.train.Saver()
        episode_step= 0
        train_step = 0
        t_start = time.time()
        date_number =0
        Track_number=0
        agent1pos=[]  #[[xt1,yt1,zt1],[xt2,yt2,zt2],....]
        agent2pos=[]
        agent3pos=[]
        Date_Number=[1]
        agent1v=[]
        agent2v=[]
        agent3v=[]
        dv1=[]
        dv2=[]
        dv3=[]        
        attacker_s=0
        defender_s=0
        f=0
        ff=0
        S_d=0.0
        S_a=0.0
        savetimes=0
        d=0
        t=0
        F=0 #追击星与目标间距离超出预定范围，任务终止
        #记录训练过程中成功率变化
        cg=[]
        theta_z=[]
        theta_xy=[]
        Done_t=[]
        obs_n = env.reset(savetimes)
       # playtime = 0
        print('Starting iterations...')
        while True:
            # get action
            action_n = [agent.action(obs) for agent, obs in zip(trainers,obs_n)]
#           =============================================================================
            #储存初始轨迹
            if arglist.savestate:  
                for i, agent in enumerate(trainers):
                    if i == 0 :
                        agent1pos.append(env.agents[i].state.p_pos.tolist())
                        agent1v.append(env.agents[i].state.p_vel.tolist())
                    if i == 1 :
                        agent2pos.append(env.agents[i].state.p_pos.tolist())
                        agent2v.append(env.agents[i].state.p_vel.tolist())                     
                date_number +=1 
#           =============================================================================
#           =============================================================================
#           博弈态势推演(逃方反应时间)
            test_step = 0
            # start_time = time.time()
            new_obs_n, rew_n, done_n, info_n, fuel_n, success_n, Fail_n,success_attacker, test_t,agent1pos,agent1v,dv1,agent2pos,agent2v,dv2,date_number = env.step(action_n,test_step,arglist.savestate,agent1pos,agent1v,dv1,agent2pos,agent2v,dv2,date_number)
            # end_time = time.time()
            # # 计算运行时间
            # run_time = end_time - start_time
            # # 打印结果
            # print("代码运行时间为：", run_time, "秒")
#           =============================================================================
            #储存控制指令
            if arglist.savestate:  
                for i, agent in enumerate(trainers):
                    if i == 0 :
                        dv1.append(env.agents[i].action.u.tolist())
                    if i == 1 :
                        dv2.append(env.agents[i].action.u.tolist())  
                if done_n:
                    Done_t.append(test_t)
            
            done = all(done_n) 
            # if done:
            #     print(done)
            episode_step += 1
            #print('追击者成功次数:',s)
            terminal = (episode_step >= arglist.max_episode_len)
            #时间耗尽或者进攻者燃料耗尽 视为进攻失败
            if done:
                d += 1
            
            if Fail_n:
                F +=1
            
            #防御者拦截成功
            elif success_n:
                defender_s +=1
                
            #进攻者成功
            elif success_attacker:
                attacker_s +=1

            elif fuel_n :
                ff +=1
            elif terminal:
                f +=1
            # 收集经验
            for i, agent in enumerate(trainers):
                agent.experience(obs_n[i], action_n[i], rew_n[i], new_obs_n[i], done_n[i], terminal)
            obs_n = new_obs_n

            for i, rew in enumerate(rew_n):
                episode_rewards[-1] += rew
                agent_rewards[i][-1] += rew



            # increment global step counter
            train_step += 1

            # for benchmarking learned policies
            if arglist.benchmark:
                for i, info in enumerate(info_n):
                    agent_info[-1][i].append(info_n['n'])
                if train_step > arglist.benchmark_iters and (done or terminal):
                    file_name = arglist.benchmark_dir + arglist.exp_name + '.pkl'
                    print('Finished benchmarking, now saving...')
                    with open(file_name, 'wb') as fp:
                        pickle.dump(agent_info[:-1], fp)
                    break
                continue

            #if len(episode_rewards) == (arglist.num_episodes):
            #    arglist.display = True 
 

            if done or terminal:
                #环境重置qian需要额外储存数据
                if arglist.savestate: 
                    for i, agent in enumerate(trainers):
                        if i == 0 :
                            agent1pos.append(env.agents[0].state.p_pos.tolist())
                            agent1v.append(env.agents[0].state.p_vel.tolist())
                            dv1.append(env.agents[0].action.u.tolist())
                        if i == 1 :
                            agent2pos.append(env.agents[1].state.p_pos.tolist())
                            agent2v.append(env.agents[1].state.p_vel.tolist())
                            dv2.append(env.agents[1].action.u.tolist())
                        if i == 2 :
                            agent3pos.append(env.agents[2].state.p_pos.tolist())
                            agent3v.append(env.agents[2].state.p_vel.tolist())
                            dv3.append(env.agents[2].action.u.tolist()) 
                    date_number +=1    
                obs_n = env.reset(savetimes)
                savetimes +=1
                episode_step = 0
                episode_rewards.append(0)
                for a in agent_rewards:
                    a.append(0)
                agent_info.append([[]])             

#           ============================================================================= 
            # for displaying learned policies
            if arglist.savestate or arglist.test or arglist.plotgj:
                if done or terminal:
                    if savetimes % 1000 ==0 or savetimes >= arglist.save_number:
                        S_d=defender_s/(1000)
                        S_a=attacker_s/(1000)   
                        theta_a1=(-90.0+10*math.floor((savetimes-1)/1000.0/36.0))
                        theta_a2=(-180.0+10*(math.floor((savetimes-1)/1000.0 %36.0)))
                        theta_z.append(theta_a1)
                        theta_xy.append(theta_a2)
                        print('theta_z:', theta_a1, 'theta_xy:', theta_a2)
                        print('拦截成功:',defender_s,'进攻成功:',attacker_s,'进攻者燃料耗尽：',ff,'时间耗尽：',f,'任务终止：',F,'任务终止2：',(1000-(defender_s+attacker_s+f+ff+F)),'进攻成功率：',S_a,'拦截成功率：',S_d,'time:',round(time.time()-t_start, 3))       
                        cg.append([defender_s,attacker_s,ff,f,F,S_d,S_a])
                        defender_s=0
                        attacker_s=0
                        ff=0
                        f=0
                        S_a=0
                        S_d=0
                    Date_Number.append(date_number)  
                if arglist.test and savetimes >= arglist.save_number:
                    book=xlwt.Workbook(encoding="utf-8")    #创建xls对象
                    worksheet=book.add_sheet("ji-li-fen-xi")  #创建一个表单 
                    col=('theta_z','theta_xy','拦截成功','进攻成功','进攻者燃料耗尽','时间耗尽','任务终止','拦截成功率','进攻成功率')                     
                    for i in range(0,9):
                        worksheet.write(0,i,col[i]) #列名  
                    if savetimes >= arglist.save_number:   
                        hangshu = math.floor(arglist.save_number/1000.0)                                
                        for i in range(0,hangshu):
                            data=[theta_z[i],theta_xy[i],cg[i][0],cg[i][1],cg[i][2],cg[i][3],cg[i][4],cg[i][5],cg[i][6]]
                            for j in range(0,9):
                                worksheet.write(i+1,j,data[j])  
                        file_name = time.strftime("机理分析DIvsBA-1.xls")
                        book.save(file_name)    #保存数据,注意必须使用xls对象操作，不能使用sheet表单操作保存  
                        break 
                    
                    
            #储存一条轨迹
            if arglist.savestate:                
                book=xlwt.Workbook(encoding="utf-8")    #创建xls对象
                worksheet=book.add_sheet("movetop")  #创建一个表单    
                col=('agent1x',"agent1y","agent1z","agent2x","agent2y","agent2z",'agent1vx',"agent1vy","agent1vz","agent2vx","agent2vy","agent2vz",'agent1dvx',"agent1dvy","agent1dvz",'agent2dvx',"agent2dvy","agent2dvz")                     
                for i in range(0,18):
                    worksheet.write(0,i,col[i]) #列名  
                if savetimes >= arglist.save_number:                                                      
                    for i in range(0,date_number):
                        data=agent1pos[i]+agent2pos[i]+agent1v[i]+agent2v[i]+dv1[i]+dv2[i]
                        for j in range(0,18):
                            worksheet.write(i+1,j,data[j])  
                    file_name = time.strftime("3d-AD1v1-2.xls")
                    book.save(file_name)    #保存数据,注意必须使用xls对象操作，不能使用sheet表单操作保存  
                    if arglist.plotgj :
                        plotguiji(Date_Number,date_number,savetimes,file_name)
                    print('储存轨迹数:',savetimes) 
                    break 



            # for displaying learned policies
            if arglist.savestate or arglist.test:  
                if  savetimes >= arglist.save_number: 
                    break
                continue
            
            # update all trainers, if not in display or benchmark mode
            loss = None
            for agent in trainers:
                agent.preupdate()
            for agent in trainers:
                loss = agent.update(trainers, train_step)

            # save model, display training output
            if terminal or done:
                                      
                if (len(episode_rewards) % arglist.save_rate == 0):  
                #if ((s+f+ff) % arglist.save_rate == 0):  
                    U.save_state(arglist.save_dir, saver=saver)
                # print statement depends on whether or not there are adversaries
                    if num_adversaries == 0:
                        print("steps: {}, episodes: {}, mean episode reward: {}, time: {}".format(
                                train_step, len(episode_rewards), np.mean(episode_rewards[-arglist.save_rate:]), round(time.time()-t_start, 3)))
                    else:
                        print("steps: {}, episodes: {}, mean episode reward: {}, agent episode reward: {}, time: {}".format(
                                train_step, len(episode_rewards), np.mean(episode_rewards[-arglist.save_rate:]),
                                [np.mean(rew[-arglist.save_rate:]) for rew in agent_rewards], round(time.time()-t_start, 3)))
                    t_start = time.time()
                #输出成功、失败的次数、成功率
                    S_d=defender_s/(1000)
                    S_a=attacker_s/(1000)                    
                    
                    print('拦截成功:',defender_s,'进攻成功:',attacker_s,'进攻者燃料耗尽：',ff,'时间耗尽：',f,'任务终止1：',F,'任务终止2：',(1000-(defender_s+attacker_s+f+ff+F)),'进攻成功率：',S_a,'拦截成功率：',S_d)                                                     

                    cg.append([defender_s,attacker_s,ff,f,F,S_d,S_a])
                    defender_s=0
                    attacker_s=0
                    f=0
                    ff=0
                    F=0
                # Keep track of final episode reward
                    final_ep_rewards.append(np.mean(episode_rewards[-arglist.save_rate:]))
                    for rew in agent_rewards:
                        final_ep_ag_rewards.append(np.mean(rew[-arglist.save_rate:]))
            
            if  (len(episode_rewards) % arglist.save_rate == 0):
                
                        #if len(episode_rewards) > arglist.num_episodes or success_n:
#                rew_file_name = arglist.plots_dir + arglist.exp_name + '_rewards.pkl'
                rew_file_name = str(arglist.plots_dir) + str(arglist.exp_name) + '_rewards.pkl'
                with open(rew_file_name, 'wb') as fp:
                    pickle.dump(final_ep_rewards, fp)
#                agrew_file_name = arglist.plots_dir + arglist.exp_name + '_agrewards.pkl'

                agrew_file_name = str(arglist.plots_dir) + str(arglist.exp_name) + '_agrewards.pkl'
                with open(agrew_file_name, 'wb') as fp:
                    pickle.dump(final_ep_ag_rewards, fp)

                cg_file_name = str(arglist.plots_dir) + str(arglist.exp_name) + '_cg.pkl'
                with open(cg_file_name, 'wb') as fp:
                    pickle.dump(cg, fp) 
    
            if len(episode_rewards) >= arglist.num_episodes:
                print('...Finished total of {} episodes.'.format(len(episode_rewards)))
                break
            

if __name__ == '__main__':

    arglist = parse_args()
    train(arglist)
