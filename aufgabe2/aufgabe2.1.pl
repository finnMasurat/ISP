:- use_module(library(clpfd)).

solve(Rows) :- append(Rows, Vars),
               Vars ins 1..9,
               maplist(all_different, Rows),
               transpose(Rows, Columns),
               maplist(all_different, Columns),
               Rows = [A,B,C,D,E,F,G,H,I],
               block(A,B,C), block(D,E,F), block(G,H,I),
               label(Vars).



block([],[],[]).
block([A,B,C | Rest1], [D,E,F | Rest2], [G,H,I | Rest3])
:- all_different([A,B,C,D,E,F,G,H,I]),
block(Rest1,Rest2,Rest3).

rows([[6,_,_,7,_,_,5,_,_],
      [_,2,8,_,_,_,_,_,_],
      [_,_,_,6,4,_,3,_,_],
      [7,4,_,_,_,_,_,2,_],
      [_,_,1,_,_,_,8,_,_],
      [_,5,_,_,_,_,_,3,7],
      [_,_,3,_,7,6,_,_,_],
      [_,_,_,_,_,_,1,9,_],
      [_,_,4,_,_,5,_,_,8]]).

do() :- rows(Rows), solve(Rows), maplist(writeln, Rows).
