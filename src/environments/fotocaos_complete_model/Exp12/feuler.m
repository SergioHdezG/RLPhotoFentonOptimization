function [t,y]=feuler(odefun,tspan,y,Nh,varargin)

%  Resuelve ecuaciones diferenciales mediantes 
%  el método de EULER progresivo.

%  [t,y]=feuler(odefun,tspan,y0,Nh) con tspan=[t0,tf]
%  integra el sistema de ecuaciones diferenciales y'=f(t,y)
%  desde el tiempo t0 hasta el tiempo tf con la condición
%  inicial y0 usando el método de Euler progresivo sobre
%  una malla uniforme de NH intervalos.
%  La función odefun(t,y) debe devolver un vector columna
%  correspondiente a f(t,y). Cada línea de la solución y 
%  corresponde a un tiempo del vector columna t.

%  [t,y]=feuler(odefun,tspan,y0,Nh,p1,p2,...) pasa los
%  parámetros adicionales p1,P2,... a la función odefun
%  escribiendo odefun(T,Y,P1,P2...).

h=(tspan(2)-tspan(1))/Nh;
tt=linspace(tspan(1),tspan(2),Nh+1);

for t = tt(1:end-1)
  y=[y;y(end,:)+...
    h*feval(odefun,t,y(end,:),varargin{:})];
end
t=tt;

return
