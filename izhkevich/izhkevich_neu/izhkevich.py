import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

class IzhikevichNeuron:
    """
    Izhikevich神经元模型实现
    """
    
    def __init__(self, a=0.02, b=0.2, c=-65, d=8):
        """
        初始化神经元参数
        
        Parameters:
        a: 恢复变量u的时间尺度参数
        b: u对v子阈值波动的敏感性
        c: 发放后膜电位重置值
        d: 发放后恢复变量u的重置增量
        """
        self.a = a
        self.b = b 
        self.c = c
        self.d = d
        
        # 状态变量
        self.v = -65   # 膜电位 (mV)
        self.u = self.b * self.v  # 恢复变量
        
        # 记录历史
        self.v_history = []
        self.u_history = []
        self.spike_times = []
        
    def reset(self):
        """重置神经元状态"""
        self.v = -65
        self.u = self.b * self.v
        self.v_history = []
        self.u_history = []
        self.spike_times = []
    
    def update(self, I, dt=0.5):
        """
        更新神经元状态一个时间步长
        
        Parameters:
        I: 输入电流
        dt: 时间步长 (ms)
        """
        # 检查是否发放动作电位
        if self.v >= 30:
            # 记录发放时间
            self.spike_times.append(len(self.v_history) * dt)
            # 复位膜电位和恢复变量
            self.v = self.c
            self.u = self.u + self.d
        else:
            # 使用欧拉法积分微分方程
            dv = (0.04 * self.v**2 + 5 * self.v + 140 - self.u + I) * dt
            du = (self.a * (self.b * self.v - self.u)) * dt
            
            self.v += dv
            self.u += du
        
        # 记录历史
        self.v_history.append(self.v)
        self.u_history.append(self.u)
        
        return self.v

def simulate_neuron(neuron_params, I, T=1000, dt=0.5, title=""):
    """
    模拟神经元并绘制结果
    
    Parameters:
    neuron_params: 神经元参数字典 {'a': , 'b': , 'c': , 'd': }
    I: 输入电流（可以是常数、数组或函数）
    T: 模拟总时间 (ms)
    dt: 时间步长 (ms)
    title: 图表标题
    """
    
    # 创建神经元实例
    neuron = IzhikevichNeuron(**neuron_params)
    
    # 时间数组
    t = np.arange(0, T, dt)
    
    # 模拟
    for i in range(len(t)):
        # 处理不同类型的输入电流
        if callable(I):
            current = I(t[i])  # I是时间函数
        elif hasattr(I, '__len__') and len(I) == len(t):
            current = I[i]     # I是数组
        else:
            current = I        # I是常数
            
        neuron.update(current, dt)
    
    # 绘制结果
    fig = plt.figure(figsize=(12, 8))
    gs = GridSpec(3, 1, figure=fig)
    
    # 膜电位图
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(t, neuron.v_history, 'b-', linewidth=1)
    ax1.set_ylabel('Membrane Potential (mV)')
    ax1.set_title(f'Izhikevich Neuron: {title}')
    ax1.grid(True, alpha=0.3)
    
    # 恢复变量图
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.plot(t, neuron.u_history, 'r-', linewidth=1)
    ax2.set_ylabel('Recovery Variable u')
    ax2.grid(True, alpha=0.3)
    
    # 输入电流图
    ax3 = fig.add_subplot(gs[2, 0])
    if callable(I):
        current_plot = [I(t_i) for t_i in t]
    elif hasattr(I, '__len__') and len(I) == len(t):
        current_plot = I
    else:
        current_plot = [I] * len(t)
    
    ax3.plot(t, current_plot, 'g-', linewidth=1)
    ax3.set_ylabel('Input Current I')
    ax3.set_xlabel('Time (ms)')
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{title}.png")
    
    # 打印发放统计信息
    print(f"{title}:")
    print(f"  Total spikes: {len(neuron.spike_times)}")
    if len(neuron.spike_times) > 1:
        isi = np.diff(neuron.spike_times)
        print(f"  Mean ISI: {np.mean(isi):.2f} ms")
        print(f"  ISI std: {np.std(isi):.2f} ms")
    print()
    
    return neuron

# 定义不同的神经元类型参数
neuron_types = {
    'Regular Spiking (RS)': {'a': 0.02, 'b': 0.2, 'c': -65, 'd': 8},
    'Intrinsically Bursting (IB)': {'a': 0.02, 'b': 0.2, 'c': -55, 'd': 4},
    'Chattering (CH)': {'a': 0.02, 'b': 0.2, 'c': -50, 'd': 2},
    'Fast Spiking (FS)': {'a': 0.1, 'b': 0.2, 'c': -65, 'd': 2},
    'Low-Threshold Spiking (LTS)': {'a': 0.02, 'b': 0.25, 'c': -65, 'd': 2},
    'Thalamo-Cortical (TC)': {'a': 0.02, 'b': 0.25, 'c': -65, 'd': 0.05},
    'Resonator (RZ)': {'a': 0.1, 'b': 0.26, 'c': -65, 'd': 2}
}

def demo_different_neurons():
    """演示不同神经元类型的发放模式"""
    
    # 恒定电流输入
    I_const = 10
    
    # 测试几种主要的神经元类型
    test_types = ['Regular Spiking (RS)', 'Intrinsically Bursting (IB)', 
                  'Fast Spiking (FS)', 'Chattering (CH)']
    
    for neuron_type in test_types:
        params = neuron_types[neuron_type]
        simulate_neuron(params, I_const, T=500, title=neuron_type)

def demo_dynamic_input():
    """演示动态输入下的神经元响应"""
    
    # 时变输入电流
    def dynamic_current(t):
        if t < 200:
            return 5
        elif t < 400:
            return 15
        else:
            return 8
    
    # 使用Regular Spiking神经元
    params = neuron_types['Regular Spiking (RS)']
    simulate_neuron(params, dynamic_current, T=600, title='RS Neuron with Dynamic Input')

def demo_poisson_input():
    """演示泊松随机输入"""
    
    T = 1000
    dt = 0.5
    t = np.arange(0, T, dt)
    
    # 创建泊松分布的随机输入
    np.random.seed(42)  # 为了可重复性
    poisson_input = np.random.poisson(0.5, len(t)) * 10
    
    params = neuron_types['Regular Spiking (RS)']
    simulate_neuron(params, poisson_input, T=T, dt=dt, title='RS Neuron with Poisson Input')

def create_network(n_neurons=100, neuron_type='RS'):
    """
    创建一个小型神经元网络演示
    """
    # 简化版网络模拟
    dt = 0.5
    T = 500
    t = np.arange(0, T, dt)
    
    # 创建神经元群体
    neurons = []
    for i in range(n_neurons):
        if neuron_type == 'RS':
            params = neuron_types['Regular Spiking (RS)']
        elif neuron_type == 'FS':
            params = neuron_types['Fast Spiking (FS)']
        else:
            params = neuron_types['Regular Spiking (RS)']
        
        neurons.append(IzhikevichNeuron(**params))
    
    # 模拟网络
    spike_trains = [[] for _ in range(n_neurons)]
    
    for time_idx in range(len(t)):
        # 基础输入 + 随机波动
        base_input = 5 + 2 * np.random.randn()
        
        for i, neuron in enumerate(neurons):
            # 简单的随机连接（这里简化处理）
            synaptic_input = 0
            for j in range(n_neurons):
                if i != j and np.random.rand() < 0.1:  # 10%的连接概率
                    if neurons[j].v > 0:  # 简化的突触后发放检测
                        synaptic_input += 0.5
            
            total_input = base_input + synaptic_input
            neuron.update(total_input, dt)
            
            # 记录发放
            if neuron.v >= 30:
                spike_trains[i].append(t[time_idx])
    
    # 绘制发放 raster 图
    plt.figure(figsize=(12, 6))
    for i in range(n_neurons):
        if spike_trains[i]:
            plt.plot(spike_trains[i], [i] * len(spike_trains[i]), 'k.', markersize=1)
    
    plt.xlabel('Time (ms)')
    plt.ylabel('Neuron Index')
    plt.title(f'Network Activity ({n_neurons} {neuron_type} Neurons)')
    plt.tight_layout()
    plt.savefig('network_activity.png')

if __name__ == "__main__":
    print("Izhikevich Neuron Model Demonstration")
    print("=" * 50)
    
    # 演示1: 不同神经元类型
    print("1. Demonstrating different neuron types...")
    demo_different_neurons()
    
    # 演示2: 动态输入
    print("2. Demonstrating dynamic input...")
    demo_dynamic_input()
    
    # 演示3: 随机输入
    print("3. Demonstrating Poisson input...")
    demo_poisson_input()
    
    # 演示4: 小型网络
    print("4. Demonstrating small network...")
    create_network(n_neurons=50, neuron_type='RS')