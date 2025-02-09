%% Extended Cahn-Hilliard Equation
nn = 511;
steps = 250;
xmin = -40;
xmax = 40;
tmax = 160;

dom = [xmin xmax]; x = chebfun('x',dom); tspan = linspace(0,tmax,steps+1);
S = spinop(dom, tspan);
S.lin = @(u) (- diff(u,2) - diff(u,4) - 0.01 * u);
S.nonlin = @(u) - diff(u - u.^3, 2); % Equivalent to -d^2/dx^2(u - u^3)
S.init = 0.9*sin((0.42)*pi*x) + 0.1*sin(10*pi*x);
u = spin(S,nn,1e-5, 'plot', 'off');

usol = zeros(nn,steps+1);
for i = 1:steps+1
    usol(:,i) = u{i}.values;
end

x = linspace(xmin,xmax,nn+1);
usol = [usol;usol(1,:)];
t = tspan;
pcolor(t,x,usol); shading interp, axis tight, colormap(jet);
usol = usol'; % shape = (steps+1, nn+1)
save('ch.mat','t','x','usol')