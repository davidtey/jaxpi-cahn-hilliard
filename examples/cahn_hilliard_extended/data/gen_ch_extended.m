%% Cahn-Hilliard Equation
dt = 5e-4;
steps = 500;
N = 511;
xmin = -1;
xmax = 1;
tmax = 4;

p = 0.5;
a = 10;
b = 9;

dom = [xmin xmax];  tspan = linspace(0,tmax,steps+1);
u0 = chebfun('0.2*(sin(10*pi*x))^3 - 0.8*sin(9*pi*x)', dom, 'trig');

% u_t = 1e-2*(-u_xx - 1e-3*u_xxxx + (u^3)_xx - u)

S = spinop(dom, tspan);
S.lin = @(u) -0.1*diff(u, 2) - 0.0001*diff(u, 4) - 0.01*u;
S.nonlin = @(u) 0.1*diff(u.^3, 2);
% S.nonlin = @(u) 0.75*u.^2.*diff(u,2) + 1.5*u.*diff(u,1).^2;
S.init = u0;
u = spin(S, N, dt, 'plot', 'off');

usol = zeros(N,steps+1);
for i = 1:steps+1
    usol(:,i) = u{i}.values;
end

x = linspace(xmin,xmax,N+1);
usol = [usol;usol(1,:)];
t = tspan;
pcolor(t,x,usol); shading interp, axis tight, colormap(jet);
usol = usol'; % shape = (steps+1, nn+1)
save('ch_extended.mat','t','x','usol')