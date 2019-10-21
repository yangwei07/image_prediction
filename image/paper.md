# 基于改进贝塞尔曲线的个性化自动驾驶决策控制研究

## 摘要
大数据时代背景下，智能汽车可利用的车辆行驶数据将多种多样，借助对自然驾驶员行驶数据的研究，智能汽车将进一步提升自动驾驶能力与行驶舒适性。本文提出改进贝塞尔曲线方法实现智能汽车安全有效地规划参考行驶路径，并建立基于高斯过程的自然驾驶员行驶速度预测模型，为路径规划碰撞安全性评价提供依据，其次，提出了结合高斯混合模型的参考路径宜人性评价策略，实现具有个性化与宜人性的自动驾驶路径规划，最后，建立最优控制二次规划模型对参考路径进行追踪，实现对参考路径的准确追踪，保证智能汽车决策系统能够从时空角度输出完整控制目标。本文所提出的路径规划与控制方法能够利用自然驾驶员行驶数据，是个性化自动驾驶的有效应用。

## 引言
随着传感器技术与计算机技术的飞速发展，智能汽车已不再局限于简单的驾驶辅助功能，而着眼于更加智能的、高效的自主控制。美国高速公路安全局报告称，“超过90%的严重车祸事故都是由驾驶员的操作失误造成”[1][2]，因此，提升车辆安全性始终是智能汽车需要面临的重点问题，与此同时，在保证车辆安全性的前提下，得益于智能汽车上高度智能化设备，汽车的各项行驶性能也具有更大的提升空间。

对于目前驾驶辅助系统而言，更高级的智能汽车应综合考虑车辆纵横向控制需求，面对时刻发生变化的动态行驶环境，更有效的决策控制策略应随着环境的变化也同时能够作出动态调整，在此方面，模型预测控制算法[3][4][5]能够有效解决动态环境的规划与控制问题，但是文献[3][4][5]中所提到的模型预测控制路径规划算法由于难以对车辆碰撞约束进行严格建模，因此，车辆在一些情况下存在较大的碰撞风险，且模型预测控制器仍然视为此种情况为有效规划，因此，这种无法建立严格碰撞约束的模型预测控制方法存在安全缺陷。文献[6][7]提出的模型预测算法能够准确建立碰撞约束，但是通常情况下这样的碰撞约束导致优化问题为非凸问题，在动态变化的行驶环境中其优化解难以保证，优化实时性也无法准确验证。因此，基于模型预测控制算法的智能汽车路径规划方法只能适用于较简单的场景，通过数值优化求解得到控制量，实现车辆自主避撞与自主变道。

不考虑车辆动力学控制问题，从环境几何建模与曲线生成的角度来研究路径规划问题是另一种常用的方法，其优势在于分别执行规划与控制，将大大简化决策控制过程，避免过于复杂的优化问题，且得益于任务分解，车辆行驶性能也能够分别考虑，进而保证安全性需求与其他性能的提升。多项式曲线[8][9]、三角函数曲线[10][11]、Logistic函数曲线[12]、贝塞尔曲线[13][14]等都常用于避撞路径规划，其路径能够通过数值表达式计算，通过动态规划能够实现较为复杂的运动路径。但是，由于动态环境复杂、驾驶员差异性大，固定参数地规划参考路径灵活性较低，并且由于驾驶员通常具有稳定的驾驶习惯，不考虑驾驶习惯进行直接规划在一定程度上将降低驾驶员舒适性。因此，考虑驾驶员特性的路径规划策略将增加舒适性考虑维度，更加全面地提升汽车智能化程度。

综上所述，本文针对高速驾驶工况下自动驾驶决策控制问题，考虑驾驶员行为特性，利用自然驾驶员行驶数据，挖掘驾驶员驾驶习惯，并应用于车辆决策规划过程，基于几何路径规划算法，通过性能设计得到一条能够保证车辆安全性且提升行驶性能的参考轨迹，并通过最优控制在追踪轨迹的同时保障行驶性能，实现智能汽车高速工况下自主行驶。

## 几何路径规划
### 三次贝塞尔曲线路径
基于几何路径规划的自动驾驶决策方法，可利用车辆运动状态与空间信息，独立生成一条用于车辆动力学控制的参考轨迹，其优势在于，将碰撞安全性独立出控制环节，仅在决策规划时考虑，因此，车辆控制器将更加关注对行驶性能的提升，实现安全、舒适、节能等综合性能需求。考虑智能汽车行驶轨迹特点，其复杂的轨迹可以由简单的几何曲线逼近得到，即在很短运动时间内，车辆的行驶轨迹可写作参数曲线，常用的参数曲线如表1所示。

|         已知条件          |    曲线类型    |                参数表达式                |                                 梯度                                  |
|:-------------------------:|:--------------:|:----------------------------------------:|:---------------------------------------------------------------------:|
|   初始位置 $(x_0, y_0)$   |    正弦曲线    |         $y = A\sin{Bx + C} + D$          |                   $\frac{dy}{dx} = AB\cos{Bx + C}$                    |
| 初始航偏角 $\frac{dy}{dx} | _{(x_0, y_0)}$ |                反正切曲线                |                      $y = A\arctan{Bx + C} + D$                       | $\frac{dy}{dx} = \frac{AB}{(Bx + C) ^ 2 + 1}$           |
|   终止位置 $(x_f, y_f)$   |  Logistic曲线  | $y = \frac{A}{1 + e ^ {-(Bx + C)} } + D$ | $\frac{dy}{dx} = \frac{ABe ^ {-(Bx + C)}}{(1 + e ^ {-(Bx + C)}) ^ 2}$ |
| 终止航偏角 $\frac{dy}{dx} | _{(x_f, y_f)}$ |               多项式曲线                | $x = A_0\tau ^ n + A_1\tau ^ {n - 1} + \cdots + A_n \\ y = B_0\tau ^ n + B_1\tau ^ {n - 1} + \cdots + B_n$  | $\frac{dy}{dx} = \frac{nB_0\tau ^ {n - 1} + \cdots +  B_{n - 1}}{nA_0\tau ^ {n - 1} + \cdots +  A_{n - 1}}$ |

可以发现，基于解析表达式的轨迹,如正弦曲线、反正切曲线、Logistic曲线，虽然能够直观判断曲线形状，但是需要联立多元方程组计算曲线参数 $A、 B、 C、 D$，在实际规划过程中不利于实时计算，而基于参数表达式的曲线由于引入中间变量，可快速得到曲线参考点。在几何路径规划问题中，通常已知条件如表1所示，利用多项式曲线求解则只能选择二次多项式得到唯一解，但是当车辆起始位置与终止位置航偏角相等时，二次多项式曲线无解。综上所述，为使路径规划实时高效并且适用范围广，则可以采用基于贝塞尔曲线差值法，其计算实时性高，并且通过车辆起止状态直接设计贝塞尔曲线控制点，实现对车辆各种状态下的有效规划。本文提出基于改进三次贝塞尔曲线的智能汽车参考轨迹规划算法，三次贝塞尔曲线规划示意图如图1所示。

![figure1](figure1.png)

根据三次贝塞尔曲线特性可知，控制点 $P_0$ 为车辆起始位置坐标 $(x_0, y_0)$，直线 $P_0P_1$ 为曲线在 $P_0$ 处的切线，即车辆的起始航偏角为

$$\tan{\theta_0} = \frac{y_{p_1} - y_{p_0}}{x_{p_1} - x_{p_0}} \tag{1}$$

同理，控制点 $P_3$ 为规划路径终点位置坐标 $(x_f, y_f)$，直线 $P_2P_3$ 为曲线在 $P_3$ 处的切线，即规划轨迹的终点航偏角为

$$\tan{\theta_f} = \frac{y_{p_3} - y_{p_2}}{x_{p_3} - x_{p_2}} \tag{2}$$

通常情况下，参考轨迹期望最终智能汽车能够稳定行驶在车道中，因此，参考轨迹的终点航偏角为 $\theta_f=0$。令车辆当前坐标为 $(x_0, y_0, \theta_0)$，所规划的轨迹终点为 $(x_f, y_f, \theta_f)$，则三次贝塞尔曲线的控制点分别为

$$P_0 = (x_0, y_0) \tag{3}$$

$$P_1 = (\frac{x_0 + x_f}{2}, y_0 + \frac{x_f - x_0}{2}tan\theta_0) \tag{4}$$

$$P_2 = (\frac{x_0 + x_f}{2}, y_f) \tag{5}$$

$$P_3 = (x_f, y_f) \tag{6}$$

$P_1$ 与 $P_2$ 的 $x$ 坐标分别取为轨迹起止点的中点是为了令转向更加平顺。然而，在某些特殊情况下，按照上述控制点规划贝塞尔轨迹会导致车辆越过道路边界的情况，如图2所示。

![figure2](figure2.png)

出现这种轨迹的原因是，车辆的规划起始航向角绝对值过大，为

$$0 < y_f - y_0 < \frac{x_f - x_0}{2} \tan{\theta_0} \ \ 或 \ \ \frac{x_f - x_0}{2} \tan{\theta_0} < y_f - y_0 < 0 \tag{7}$$

在上式中，由于起止点姿态信息已知，上式（7）分别对应航向角为正、负两种情况，在此情况下，需对贝塞尔曲线轨迹的控制点进行修正，如图2中虚线所示，$P_1$ 控制点的 $y$ 坐标应在$P_0$与$P_2$之间，为便于计算，取 $P_1$ 的坐标为
$$
P_1 =
\begin{cases}
    (x_0 + \frac{y_f - y_0}{\tan{\theta_0}}, y_f) & 满足公式(7)，需修正\\
    \\
    (\frac{x_0 + x_f}{2}, y_0 + \frac{x_f - x_0}{2} \tan{\theta_0}) & 不满足公式(7)，不需修正
\end{cases} \tag{8}
$$
最终，通过对贝塞尔曲线控制点的修正，实现所有条件下都能有效规划。三次贝塞尔曲线为

$$P = (1 - \tau) ^ 3 P_0 + 3(1 - \tau) ^ 2 \tau P_1 + 3(1 - \tau) \tau ^ 2 P_2 + \tau ^ 3 P_3 \tag{9}$$
通过在道路坐标系下离散采样可以得到若干备选终止点，如图3所示， 最终通过终止点采样可以生成一簇备选参考轨迹。

![figure3](figure3.png)

### 障碍车运动状态预测
保证行驶安全，即车辆严格跟踪所规划的轨迹时不会发生碰撞是参考路径选取的基本要求，由于障碍车运动具有随机性，通常采用的线性运动预测将产生较大误差，本文假设障碍车运动符合高斯过程，即障碍车的运动状态在连续时域内符合多元高斯分布，协方差函数确定了不同时刻的运动状态相关性，通过对障碍车运动历史数据的学习，利用多元高斯分布可预测一定时域内障碍车的纵向运动速度[15]。

假设观测到 $x_0$ 到 $x_n$ 时刻过程中，障碍车的纵向速度为 $\mathbf{y} = \{ y_0, y_1, \cdots, y_n \}$，其中，时间记作 $\mathbf{x} = \{ x_0, x_1, \cdots, x_n \}$，预测时间为 $\mathbf{x_\ast}$，相应的预测状态为 $\mathbf{y_\ast}$，则观测状态可转化一个零均值的多元高斯分布，其先验分布为
$$
\left[\begin{matrix}
    \mathbf{y} \\
    \mathbf{y_\ast} \\
\end{matrix} \right] \sim N(0, \Sigma) \tag{10}
$$
$$
\Sigma = \left[\begin{matrix}
        K & K_\ast ^ T \\
        K_\ast & K_{\ast\ast} \\
        \end{matrix} \right]) \tag{11}
$$
其中，$K, K_\ast, K_{\ast\ast}$ 分别为多元协方差矩阵，由协方差函数 $k(x, x')$ 构成，为
$$
K = \left[\begin{matrix}
k(x_0, x_0) & k(x_0, x_1) & \cdots & k(x_0, x_n)\\
k(x_1, x_0) & k(x_1, x_1) & \cdots & k(x_1, x_n)\\
\vdots & \vdots & \ddots & \vdots \\
k(x_n, x_0) & k(x_n, x_1) & \cdots & k(x_n, x_n)
\end{matrix}\right] \tag{12}
$$

$$
K_\ast = \left[ \begin{matrix} k(\mathbf{x_\ast}, x_0) & k(\mathbf{x_\ast}, x_1) & \cdots & k(\mathbf{x_\ast}, x_n)\end{matrix} \right] \tag{13}
$$

$$K_{\ast\ast} = k(\mathbf{x_\ast}, \mathbf{x_\ast}) \tag{14}$$

在此，本文选取平方指数函数作为协方差函数为

$$ k(x, x') = \sigma_f ^2 \exp{[-\frac{(x-x') ^ 2}{2l ^ 2}]} \tag{15}$$

其中，$\theta = (\sigma_f, l)$ 是待求解的超参数，可通过极大似然估计求解，

$$\theta = \mathop{\arg\min}_{\theta} \ -\sum{\log{p(\mathbf{y}|\mathbf{x}, \theta)}} = \mathop{\arg\min}_{\theta} \ -\sum{[\frac{1}{2} \log{|K|} + \frac{1}{2}\mathbf{y} ^ T K ^ {-1} \mathbf{y} + \frac{n}{2} \log{(2\pi)}]} \tag{16}$$

最终，障碍车的预测状态期望与方差由条件分布得到，为

$$\mathbf{y_\ast}|\mathbf{y} \sim N(K_\ast K ^ {-1}\mathbf{y}, K_{\ast\ast} - K_\ast K ^ {-1}K_\ast ^ T) \tag{17}$$

本文基于西班牙阿尔卡拉大学公开自然驾驶数据库[16]对障碍车状态进行预测，其预测结果如图4所示

![figure4](figure4.png)

采样数据间隔为0.5秒，通过5秒的历史数据对未来2秒的障碍车车速进行预测，通过极大似然估计实时学习车辆5秒车速历史数据，得到高斯过程最优超参数，进而可知障碍车未来车速随预测时间的均值与置信区间，该方法给出了障碍车随机运动的速度概率分布，通过置信区间得到在稳定行驶状态下障碍车的车速区间，实现对障碍车运动预测。针对图4模拟预测的高斯过程模型，其超参数分别为

| 预测时间段  | $\sigma_f$ | $l$  |
|:-----------:|:----------:| :----: |
| prediction0 |    1.53    | 3.44 |
| prediction1 |    1.44    | 1.50 |
| prediction2 |    1.01    | 1.86 |
| prediction3 |    1.88    | 2.66 |
| prediction4 |    2.01    | 2.26 |

由公式（18）可知，在训练区间，预测区间，采样间隔确定后，可以得到速度预测最大方差，即预测不确定性大小，为

$$\sigma_{max} ^ 2(\sigma_f, l) = \{K_{\ast\ast} - K_\ast K ^ {-1}K_\ast ^ T |\mathbf{x} = 0, 0.5, \cdots, 4.5, 5, \mathbf{x_\ast}=7 \} \tag{18}
$$

其受超参数 $\theta = (\sigma_f, l)$ 影响，如图5所示

![figure5](figure5.png)

### 安全性约束
通过对障碍车运动状态预测，智能汽车在规划路径时可以实时判断所规划的路径是否存在碰撞风险，在此碰撞条件由图6所示

![figure6](figure6.png)

假设车辆以包围它的椭圆范围内为安全区域，且始终以智能汽车中心位置作为坐标系原点，车辆纵向方向作为坐标系横轴，则安全性约束为

$$C: \frac{s_x ^ 2}{\Delta_x} + \frac{s_y ^ 2}{\Delta_y} \geq 1 \tag{19}$$

其中，$s_x$ 与 $s_y$ 分别为智能汽车传感器探测到的障碍物最近点到坐标系原点的纵向与横向距离，$\Delta_x$ 与 $\Delta_y$ 分别为车辆纵向安全裕度与横向安全裕度，即安全椭圆的半长轴与半短轴。通过对障碍物运动预测及安全约束的建立，不满足安全性要求的备选轨迹将舍弃，留下能够保障车辆安全的若干条备选轨迹。

### 最优参考轨迹
在满足安全性约束条件下，每一条规划的路径均可作为智能汽车安全自动驾驶的参考轨迹，但是由于轨迹不同，所表现出的驾驶性能也存在差异，此时，通过对性能目标的设计与优化，最终可以得到一条既能保证车辆行驶安全又能够获得较好行驶性能的自动驾驶参考路径。在决策层面，本文主要从以下两个方面考虑行驶性能：（1）宜人性：在纵向运动控制中，所规划的轨迹应符合驾驶员的驾驶习惯与驾驶倾向；（2）合理性：车辆尽量行驶在车道中央，且通常以内侧车道作为超车道.

针对宜人性，首先需要学习驾驶员跟车驾驶习惯，同样基于文献[16]中的数据，建立高斯混合模型学习驾驶员行驶特性，其中，高斯混合模型为

$$f(X|\phi_i, \mu_i, \Sigma_i) = \sum_i ^ k{\phi_i \cdot N(X|\mu_i, \Sigma_i)} \tag{20}$$

$$N(X|\mu_i, \Sigma_i) = \frac{1}{2\pi} \sqrt{\frac{1}{\det{|\Sigma_i|}}} \exp{[-\frac{1}{2}(X - \mu_i) ^ T \Sigma_i ^ {-1} (X - \mu_i)]} \tag{21}$$

采用EM算法[17]学习高斯混合模型如图7，最优高斯混合参数如表3所示。

![figure7](figure7.png)

| 高斯成分 $i$ | 权重 $\phi$ |       均值 $\mu$      |                                    协方差 $\Sigma$                                   |
|:--------:|:----:|:----------------:|:----------------------------------------------------------------------------:|
|    1     | 0.11 | $[26.24\ -0.42]$ | $\left[\begin{matrix} 69.85 & -16.81 \\ -16.81 & 285.84 \end{matrix}\right]$ |
|    2     | 0.44 | $[24.88\ -0.65]$ | $\left[\begin{matrix} 41.48 & -1.13 \\ -1.13 & 8.82 \end{matrix}\right]$     |
|    3     | 0.45 | $[17.05\ -0.18]$ | $\left[\begin{matrix} 4.17 & 0.14 \\ 0.14 & 1.56 \end{matrix}\right]$        |

通过对驾驶员跟车工况下跟车距离-相对速度的特性建模，评估所生成的三次贝塞尔曲线路径的宜人特性，其归一化的后验概率密度越大，说明所生成路径的宜人性越高，越符合所学习的驾驶员特性，即

$$J_1 = \sum{\log{p(s, v|\phi, \mu, \Sigma)}} \tag{22}$$

其中，$s$ 表示所规划路径中，本车与障碍物的相对距离，$v$ 表示在规划时域一定情况下，本车与障碍物的相对速度，$\phi, \mu, \Sigma$ 为高斯混合模型参数。

针对合理性，每条车道将赋予不同的权重以及每条车道中线的权重也将更大

$$J_2 = \sum{(y-y_{target}) ^ 2} \tag{23}$$

综上所述，针对车辆行驶性能优化设计为

$$\min \ \ J = \omega_1J_1 + \omega_2J_2 \tag{25}$$

其中，$\omega$ 为权重因子。最终，通过对性能目标的择优选取，能够生成一条满足安全性，并能更优发挥车辆性能的参考路径，如图8所示。

![figure8](figure8.png)

## 基于最优控制的路径追踪策略
基于所提出的三次贝塞尔曲线路径规划方法，能够生成得到一条用于控制器追踪控制的参考路径，在规划过程中已考虑车辆安全性要求，但车辆性能优化仅仅是规划要求，实际车辆性能发挥必须通过控制器实现，因此，追踪控制的目标主要是减少轨迹追踪误差以及最终实现车辆性能发挥。针对此类问题，最优控制是实现追踪与性能优化的有效途径。

建立三自由度车辆动力学模型如图9所示

![figure9](figure9.png)

$$\dot{v_T} = \frac{1}{m}[F_{xF}\cos{(\alpha_T - \delta)} + F_{xR}\cos{\alpha_T} + F_{yF}\sin{(\alpha_T - \delta)} + F_{yR}\sin{\alpha_T}] \tag{26}$$

$$\dot{\alpha_T} = \frac{1}{mv_T}[-F_{xF}\sin{(\alpha_T - \delta)} - F_{xR}\sin{\alpha_T} + F_{yF}\cos{(\alpha_T - \delta)} + F_{yR}\cos{\alpha_T}] - \dot{\psi} \tag{27}$$

$$\ddot{\psi} = \frac{1}{I}[aF_{xF}\sin{\delta} + aF_{yF}\cos{\delta} - bF_{yR}] \tag{28}$$

坐标系状态微分方程为

$$\dot{x} = v_T\cos{(\theta + \alpha_T)} \tag{29}$$

$$\dot{y} = v_T\sin{(\theta + \alpha_T)} \tag{30}$$

$$\dot{\theta} = \dot{\psi} \tag{31}$$

离散并线性化状态空间方程为

$$\mathbf{\dot{x} = Ax + Bu} \tag{32}$$

$$\mathbf{y = C(x - x_{ref})} \tag{33}$$

其中,

$$\mathbf{x} = [x, y, \theta, v_T, \alpha_T, \dot{\psi}] ^ T \tag{34}$$

$$
\mathbf{A} = \left[\begin{matrix}
0 & 0 & 0 & 1 & 0 & 0\\
0 & 0 & v_T & 0 & v_T & 0\\
0 & 0 & 0 & 0 & 0 & 1\\
0 & 0 & 0 & 0 & 0 & 0\\
0 & 0 & 0 & 0 & \frac{K_F + K_R}{mv_T} & 1 + \frac{aK_F - bK_R}{mv_T ^ 2}\\
0 & 0 & 0 & 0 & \frac{aK_F - bK_R}{I} & \frac{a ^ 2K_F + b ^ 2K_R}{Iv_T}\\
\end{matrix} \right] \tag{35}
$$

$$
\mathbf{B} = \left[\begin{matrix}
0 & 0 & 0\\
0 & 0 & 0\\
0 & 0 & 0\\
0 & \frac{1}{m} & \frac{1}{m}\\
\frac{K_F}{mv_T} & 0 & 0\\
\frac{aK_F}{I} & 0 & 0\\
\end{matrix}\right] \tag{36}
$$

$$
\mathbf{C} = \left[\begin{matrix}
0 & 1 & 0 & 0 & 0 & 0\\
0 & 0 & 1 & 0 & 0 & 0\\
0 & 0 & 0 & 1 & 0 & 0\\
0 & 0 & 0 & 0 & 1 & 0\\
0 & 0 & 0 & 0 & 0 & 1\\
\end{matrix}\right] \tag{37}$$

$$\mathbf{u} = [\delta, F_{xF}, F_{x_R}] ^ T \tag{38}$$

最优控制器设计为

$$
\mathop{\arg\min}_{\mathbf{u}} \ \ \mathbf{\int_{0}^{t_f}{(y ^ TQy + u ^ TRu)dt}}\\
s.t.
\begin{cases}
    \mathbf{\dot{x} = Ax + bu}\\
    \mathbf{y = C(x - x_{ref})}\\
    \mathbf{u \in u ^ D}\\
    \mathbf{y \in y ^ D}
\end{cases}\tag{39}$$

本文在文献[16]的数据基础上模拟了高速公路自动驾驶场景，障碍车为自然驾驶员行驶的车辆，而主车为自动驾驶汽车，其运动控制仿真如图10所示。

![figure10](figure10.png)
![figure11](figure11.png)

图10（b）为采用最优控制求解得到车辆侧向跟踪误差，其精度足以满足自动驾驶车辆轨迹追踪要求。由图10可知，本文提出的路径规划与控制算法能够有效实现自动驾驶规划，并且由于每个规划步长是由贝塞尔曲线构成，在长期自动驾驶过程中，能够实现复杂的曲线运动以及对动态障碍物的实时避让，保证了行驶安全性。由于最优控制目标不包含非线性的碰撞约束，优化问题为二次规划，保证了最优控制有解，即所规划的路径能够追踪实现。

## 结论
本文以“规划-控制”架构建立了自动驾驶车辆运动决策系统，以车辆安全性为目标设计基于贝塞尔曲线的路径规划算法，并考虑驾驶员特性，建立高斯混合模型评估规划路径宜人性。其次，基于驾驶员行驶数据，本文建立了基于高斯过程的自然驾驶员车速预测模型，为路径规划碰撞约束提供依据。最后，设计最优控制器确定车辆控制参数，保证决策系统能够输出完整车辆追踪控制目标。本文研究主要有以下几点创新之处：
1）基于高斯过程模型预测自然驾驶人构成的障碍车运动速度，实现在路径规划时能够更加准确地预判路径安全性；
2）通过自然驾驶员行驶数据建立驾驶员特性模型，实现了对所规划的路径宜人性评价，使得自动驾驶纵向运动路径更加符合驾驶员驾驶特性；
3）最优控制只考虑对追踪目标的精度，路径规划时通过碰撞条件筛选安全路径，保证了最优控制为二次规划，有效解存在。
本文为自动驾驶宜人性设计提供初步依据，实现了对自然驾驶员行驶数据的利用，并考虑驾驶员特性提升自动驾驶路径规划的合理性与宜人性，并且路径规划算法以采样计算为核心，具有较大的工程应用潜力。
## 参考文献
[1] Singh, Santokh. Critical reasons for crashes investigated in the national motor vehicle crash causation survey. No. DOT HS 812 115. 2015.\
[2] Huh, Kunsoo, Chanwon Seo, Joonyoung Kim, and Daegun Hong. "An experimental investigation of a CW/CA system for automobiles using hardware-in-the-loop simulations." In American Control Conference, 1999. Proceedings of the 1999, vol. 1, pp. 724-728. IEEE, 1999.\
[3] Rasekhipour, Yadollah, Amir Khajepour, Shih-Ken Chen, and Bakhtiar Litkouhi. "A potential field-based model predictive path-planning controller for autonomous road vehicles." IEEE Transactions on Intelligent Transportation Systems 18, no. 5 (2017): 1255-1267.\
[4] MPC\
[5] MPC\
[6] Du, Yaoqiong, Yizhou Wang, and Ching-Yao Chan. "Autonomous lane-change controller via mixed logical dynamical." In Intelligent Transportation Systems (ITSC), 2014 IEEE 17th International Conference on, pp. 1154-1159. IEEE, 2014.\
[7] Liu, Chang, Seungho Lee, Scott Varnhagen, and H. Eric Tseng. "Path planning for autonomous vehicles using model predictive control." In 2017 IEEE Intelligent Vehicles Symposium (IV), pp. 174-179. IEEE, 2017.\
[8] Yang, Da, Shiyu Zheng, Cheng Wen, Peter J. Jin, and Bin Ran. "A dynamic lane-changing trajectory planning model for automated vehicles." Transportation Research Part C: Emerging Technologies 95 (2018): 228-247.\
[9] Heil, Thomas, Alexander Lange, and Stephanie Cramer. "Adaptive and efficient lane change path planning for automated vehicles." In Intelligent Transportation Systems (ITSC), 2016 IEEE 19th International Conference on, pp. 479-484. IEEE, 2016.\
[10] Ghommam, Jawhar, Hasan Mehrjerdi, Maarouf Saad, and Faiçal Mnif. "Formation path following control of unicycle-type mobile robots." Robotics and Autonomous Systems 58, no. 5 (2010): 727-736.\
[11] Chen, Hongda, Kuochu Chang, and Craig S. Agate. "UAV path planning with tangent-plus-Lyapunov vector field guidance and obstacle avoidance." IEEE Transactions on Aerospace and Electronic Systems 49, no. 2 (2013): 840-856.\
[12] Upadhyay, Saurabh, and Ashwini Ratnoo. "Continuous-curvature path planning with obstacle avoidance using four parameter logistic curves." IEEE Robotics and Automation Letters 1, no. 2 (2016): 609-616.\
[13] Qian, Xiangjun, Inaki Navarro, Arnaud de La Fortelle, and Fabien Moutarde. "Motion planning for urban autonomous driving using Bézier curves and MPC." In Intelligent Transportation Systems (ITSC), 2016 IEEE 19th International Conference on, pp. 826-833. Ieee, 2016.\
[14] Choi, Ji-wung, Renwick Curry, and Gabriel Elkaim. "Path planning based on bézier curve for autonomous ground vehicles." In World Congress on Engineering and Computer Science 2008, WCECS'08. Advances in Electrical and Electronics Engineering-IAENG Special Edition of the, pp. 158-166. IEEE, 2008.\
[15] Ngo, Phillip, Wesam Al-Sabban, Jesse Thomas, Will Anderson, Jnashewar Das, and Ryan N. Smith. "An analysis of regression models for predicting the speed of a wave glider autonomous surface vehicle." In Proceedings of Australasian Conference on Robotics and Automation. Australian, pp. 1-9. 2013.\
[16] Romera, Eduardo, Luis M. Bergasa, and Roberto Arroyo. "Need data for driver behaviour analysis? Presenting the public UAH-DriveSet." In 2016 IEEE 19th International Conference on Intelligent Transportation Systems (ITSC), pp. 387-392. IEEE, 2016.\
[17] McLachlan, Geoffrey, and Thriyambakam Krishnan. The EM algorithm and extensions. Vol. 382. John Wiley & Sons, 2007.\
