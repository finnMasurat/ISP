:- use_module(library(clpfd)).

sudoku(Rows) :-
        append(Rows, Vs),                   % Konkatination von Rows in Vs. True wenn Rows eine Liste von Vs ist
        Vs ins 1..9,                        % Vs sind Elemente von von 1..9
        maplist(all_different, Rows),       % True wenn all_different auf Rows angewendet werden kann
        transpose(Rows, Columns),
        maplist(all_different, Columns),    % Transponiert Matrix für den Aufbau
        Rows = [A,B,C,D,E,F,G,H,I],         % Rows mit Variablen
        blocks(A, B, C),                    %
        blocks(D, E, F),                    %
        blocks(G, H, I),                    %
        label(Vs).                          % Labeling der Liste

blocks([], [], []).
blocks([A,B,C|Bs1], [D,E,F|Bs2], [G,H,I|Bs3]) :-
        all_different([A,B,C,D,E,F,G,H,I]),
        blocks(Bs1, Bs2, Bs3).

problem([[_,_,4,7,_,8,_,1,_],
          [_,_,7,_,_,1,3,4,_],
          [_,_,_,_,3,_,_,_,5],
          [_,_,1,_,_,6,_,_,_],
          [3,_,_,_,_,_,_,_,7],
          [_,_,_,8,_,_,9,_,_],
          [2,_,_,_,9,_,_,_,_],
          [_,1,3,5,_,_,7,_,_],
          [_,9,_,6,_,3,4,_,_]]).

do() :- problem(Rows), sudoku(Rows), maplist(writeln, Rows).
