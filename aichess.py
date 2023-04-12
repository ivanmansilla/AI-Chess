#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  8 11:22:03 2022

@author: ignasi
"""
import copy

import chess
import numpy as np
import sys
import queue
import math
from typing import List

RawStateType = List[List[List[int]]]
import sys
from itertools import permutations
import random
import json


class Aichess():
    """
    A class to represent the game of chess.

    ...

    Attributes:
    -----------
    chess : Chess
        represents the chess game

    Methods:
    --------
    startGame(pos:stup) -> None
        Promotes a pawn that has reached the other side to another, or the same, piece

    """

    def __init__(self, TA, myinit=True):
        limit = sys.getrecursionlimit()
        sys.setrecursionlimit(2000)
        if myinit:
            self.chess = chess.Chess(TA, True)
        else:
            self.chess = chess.Chess([], False)

        self.listNextStates = []
        self.listVisitedStates = []
        self.pathToTarget = []
        self.currentStateW = self.chess.boardSim.currentStateW;
        self.depthMax = 1300;
        self.finaldepth = 0;
        self.checkMate = False

    def getCurrentState(self):

        return self.myCurrentStateW

    def getListNextStatesW(self, myState):

        self.chess.boardSim.getListNextStatesW(myState)
        self.listNextStates = self.chess.boardSim.listNextStates.copy()

        return self.listNextStates

    def getListNextStatesB(self, myState):

        self.chess.boardSim.getListNextStatesB(myState)
        self.listNextStates = self.chess.boardSim.listNextStates.copy()

        return self.listNextStates

    def isSameState(self, a, b):

        isSameState1 = True
        # a and b are lists
        for k in range(len(a)):

            if a[k] not in b:
                isSameState1 = False

        isSameState2 = True
        # a and b are lists
        for k in range(len(b)):

            if b[k] not in a:
                isSameState2 = False

        isSameState = isSameState1 and isSameState2
        return isSameState

    def isVisited(self, mystate):

        if (len(self.listVisitedStates) > 0):
            perm_state = list(permutations(mystate))

            isVisited = False
            for j in range(len(perm_state)):

                for k in range(len(self.listVisitedStates)):

                    if self.isSameState(list(perm_state[j]), self.listVisitedStates[k]):
                        isVisited = True

            return isVisited
        else:
            return False

    def overLayer(self, state):
        """
        - Indica solapament entre peçes negres / rei blanc
        Args:1
            state: Posició peçes

        Returns: bool

        """
        if (state[0][0] == state[1][0] and state[0][1] == state[1][1]):
            return True

        if (state[0][0] == 0 and state[0][1] == 4 or state[1][0] == 0 and state[1][1] == 4):
            return True

        return False

    def moveWhite(self, state):
        """
        - Realitza moviment peçes blanques
        - Actualitza board
        Args:
            state: Posició peçes objectiu
        Returns: None
        """
        TA = np.zeros((8, 8))
        TA[0][4] = 12
        TA[state[0][0]][state[0][1]] = state[0][2]
        TA[state[1][0]][state[1][1]] = state[1][2]
        self.chess = chess.Chess(TA, True)

    def isCheckMate(self, mystate):
        """
        - Comprovacio de checkmate donat un state
        Args:
            mystate: Possible estat check mate

        Returns: bool
            - Si hi ha check mate
        Exceptions:
            - Adjaçents al rei no es checkmate( rei podria menjar la torre)
            - Torre no es superposa amb el rei negre
        """
        # 2 condicions a complir
        ismateKing = False
        ismateRook = False

        # Excepcions
        notmate = [3, 4, 5]
        for s in mystate:
            if s[2] == 2 and s[0] == 0:
                if s[1] not in notmate:
                    ismateRook = True

            if s[2] == 6 and s[0] == 2 and s[1] == 4:
                ismateKing = True
            if ismateKing and ismateRook:
                self.moveWhite(mystate)
                #aichess.chess.boardSim.print_board()

        return ismateKing and ismateRook

    def DepthFirstSearch(self, currentState, depth):
        """
        - Cerca DFS
        Args:
            currentState: Posició peçes per explorar
            depth: profunditat
        Returns: bool
            - indica primer check mate
        """
        self.listVisitedStates.append(currentState)
        # Llindar depth i solapaments
        if self.overLayer(currentState) or depth >= self.depthMax:
            return False
        # CheckMate
        if (self.isCheckMate(currentState)):
            self.pathToTarget.append(currentState)
            return True
        # Actualitzem taulell
        depth += 1
        self.moveWhite(currentState)
        nextstates = self.getListNextStatesW(currentState)

        for n in nextstates:
            # Funcio donada de control
            if self.isSameState(n[0], n[1]):
                nextstates.remove(n)
                return False
            if not (self.isVisited(n)):
                self.finaldepth = depth
                if (self.DepthFirstSearch(n, depth)):
                    self.pathToTarget.append(currentState)
                    return True

        return False

    def BreadthFirstSearch(self, currentState):
        depth = 0
        if self.overLayer(currentState):
            return 0
        # Afegim el primer estat com visitat
        self.listVisitedStates.append(currentState)
        # Creem la queue on aniran els estats que anirem comprovant
        queue2 = queue.Queue()
        queue2.put(currentState)
        parent = dict()
        # El primer node no te pare
        parent = {str(currentState): None}
        # Hem de comprovar que el primer node no sigui ja checkMate
        if (self.isCheckMate(currentState)):
            # for move in parent:
            #    self.pathToTarget.append(move)
            return True
        # Visitem cada node
        while (queue2):
            # Estat que estem mirant
            actualState = queue2.get()
            depth += 1
            # Comprovem si es checkMate
            if (self.isCheckMate(actualState)):
                # Comentem la part d'afegir al pathToTarget, per problemes amb el diccionari
                """move = actualState
                while (parent[str(move)] != None):
                    for key, value in parent:
                        if (value == move):
                            move = parent[key]
                            self.pathToTarget.append(move)
                            parent.pop(move)"""
                self.chess.boardSim.print_board()
                return True
            # Actualitzem taular i llista dels nextStates
            self.moveWhite(actualState)
            nextStates = self.getListNextStatesW(actualState)
            # Mirem els posibles moviments
            for neighbour in nextStates:
                # Actualitzem pare
                if not (self.overLayer(neighbour)):
                    parent[str(neighbour)] = actualState
                    # Per a que sigui mes eficient, evitem que estiguin a la mateixa posicio
                    if (self.isSameState(neighbour[0], neighbour[1])):
                        nextStates.remove(neighbour)
                    else:
                        # Si no ha estat visitat, l'afegim a la queue i a la llista de visitats
                        if not (self.isVisited(neighbour)):
                            self.finaldepth = depth
                            self.listVisitedStates.append(neighbour)
                            queue2.put(neighbour)

    def isCheckMate2(self, mystate):
        ADJACENTS = {(-1, 1), (0, 1), (1, 1), (-1, 0),
                     (1, 0), (-1, -1), (0, -1), (1, -1), (0, 0)}
        status = False
        board = np.zeros((8, 8), dtype=int)
        r = range(0, 8)
        for pos in mystate:
            board[pos[0], pos[1]] = 1
            if pos[2] == 2:
                board[pos[0], :] = 1
            board[:, pos[1]] = 1
            if pos[2] == 6:
                for dy, dx in ADJACENTS:
                    print(dx, dy)
                    if (pos[0] + dx in r) and (pos[1] + dy in r):
                        print('dins')
                        print(pos[0] + dy)
                        print(pos[1] + dx)
                        board[pos[0] + dy, pos[1] + dx] = 1
            if pos[2] == 0:
                r = range(0, 8)
                for dy, dx in ADJACENTS:
                    if (pos[0] + dx in r) and (pos[1] + dy in r):
                        if (board[pos[0] + dy, pos[1] + dx] != 1):
                            status = True
                        else:
                            status = False
        print(board)
        return status

    def moveMinMax(self, stateW, stateB):
        """
        - Realitza moviment peçes blanques
        - Actualitza board
        Args:
            state: Posició peçes objectiu
        Returns:
            None
        """
        TA = np.zeros((8, 8))
        TA[stateW[0][0]][stateW[0][1]] = stateW[0][2]
        TA[stateW[1][0]][stateW[1][1]] = stateW[1][2]
        TA[stateB[0][0]][stateB[0][1]] = stateB[0][2]
        TA[stateB[1][0]][stateB[1][1]] = stateB[1][2]
        virtualchess = Aichess(TA, True)
        # virtualchess.chess.boardSim.print_board()
        return virtualchess

    def kingRook(self, currentStateW, currentStateB, maximizingPlayer):
        ADJACENTS = {(-1, 1), (0, 1), (1, 1), (-1, 0),
                     (1, 0), (-1, -1), (0, -1), (1, -1), (0, 0)}
        if (len(currentStateB) < 2 or len(currentStateW) < 2):
            if (maximizingPlayer):
                value = math.inf
            else:
                value = -math.inf
            return value
        for i in currentStateW:
            if (i[2] == 2):
                posRW = i
            if (i[2] == 6):
                posKW = i
        for j in currentStateB:
            if (j[2] == 12):
                posKB = j
            if (j[2] == 8):
                posRB = j

        maxP = 0
        minP = 0
        rr = False
        rk = False

        # Rook check
        print(currentStateW, currentStateB)
        if (posRW[0] == posRB[0] or posRW[1] == posRB[1]):
            rr = True
            if (maximizingPlayer):
                maxP = -math.inf
            else:
                minP = math.inf

        # Check King Black
        if (posRW[0] == posKB[0] or posRW[1] == posKB[1]):
            if (maximizingPlayer):
                maxP = max(maxP, 1)
            else:
                minP = math.inf
            rk = True

        # Check King White
        if (posRB[0] == posKW[0] or posRB[1] == posKW[1]):
            if (maximizingPlayer):
                maxP = -math.inf
            else:
                minP = min(minP, -1)
            rk = True

        # WRook surrounding BKing
        for dy, dx in ADJACENTS:
            if (posKB[0] + dx == posRW[0]) and (posKB[1] + dy == posRW[1]):
                maxP = -math.inf

        # WKing surrounding BKing
        for dy, dx in ADJACENTS:
            if (posKB[0] + dx == posKW[0]) and (posKB[1] + dy == posKW[1]):
                maxP = -math.inf
                minP = +math.inf

        # Check rook-rook & rook-king
        if (rk and rr):
            if (posKB[1] > posRB[1] and posKB[1] < posRW[1]):
                mapP = 1
                minP = math.inf
            elif (posKB[1] < posRB[1] and posKB[1] > posRW[1]):
                mapP = 1
                minP = math.inf
            elif (posKB[0] > posRB[0] and posKB[0] < posRW[0]):
                mapP = 1
                minP = math.inf
            elif (posKB[0] < posRB[0] and posKB[0] > posRW[0]):
                mapP = 1
                minP = math.inf
            else:
                maxP = -math.inf
                minP = +math.inf

        # Capture Rook-Rook
        if (posRW[0] == posRB[0] and posRW[1] == posRB[1]):
            if (maximizingPlayer):
                maxP = math.inf
            else:
                minP = -math.inf

        # Return depending player
        if (maximizingPlayer):
            return maxP
        else:
            return minP

    def minimax(self, currentStateW, currentStateB, depth, maximizingPlayer, moviment):
        boardMinMax = self.moveMinMax(currentStateW, currentStateB)
        if (depth == 0):
            value = aichess.kingRook(currentStateW, currentStateB, not maximizingPlayer)
            print(currentStateW, "i el seu value", value)
            moviment[str(currentStateW)] = value
            return value

        if (maximizingPlayer):
            # Value = cridar funcio per saber el valor
            value = -math.inf
            boardMinMax = self.moveMinMax(currentStateW, currentStateB)
            nextStatesW = boardMinMax.getListNextStatesW(currentStateW)  # self.getListNextStatesW(currentStateW)
            for child in nextStatesW:
                depth2 = depth - 1
                if self.overLayer(child):
                    continue
                value = max(value, aichess.minimax(child, currentStateB, depth2, False, moviment))
            max_value = max(moviment.values())
            movFinal = [k for k, v in moviment.items() if v == max_value]
            print("MAXI: Chld amb major value", movFinal, " amb un valor de: ", max_value)
            # print("FINALL")
            return value
        else:  # El minimizingPlayer
            # Value = cridar funcio per saber el valor
            # value = aichess.setValue(self, node)
            value = math.inf
            boardMinMax = self.moveMinMax(currentStateW, currentStateB)
            nextStatesB = boardMinMax.getListNextStatesB(currentStateB)  # self.getListNextStatesB(currentStateB)
            for child2 in nextStatesB:
                depth2 = depth - 1
                value = min(value, aichess.minimax(currentStateW, child2, depth2, True, moviment))
            min_value = min(moviment.values())
            movFinal = [k for k, v in moviment.items() if v == min_value]
            print("MINI: Chld amb major value", movFinal, " amb un valor de: ", min_value)
            return value

        # Diccionari-> Key: Moviment, Value: un diccionari amb possibles moviments, el valor, i el millor possibile moviment

    def valueState(self, currentState, allMoves, list_moves):
        """
            - Funció per construir o actualitzar el value de cada estat on esta la peça(key). El value esta format per:
            el valor de cada estat, i un diccionari on tindrem els possibles següents moviments (key) i el seu valor (value)
            Args:
                currentState: Poisció del estat on està la peça actualment
                allMoves: Diccionari principal, on hi ha tots els estats on ha estat la peça amb els seus values i
                possibles moviments
                list_moves: Proxims possibles moviments a fer desde la posició actual
            Returns: allMoves[sCurrentState]  -> Value construit o actualitzat del estat actual dins el dicc principal
        """
        sCurrentState = str(currentState)
        # Comprovem si l'estat actual dins el diccionar allMoves per tal de construir-ho si cal
        # Si no hi hes vol dir que la peça encara no ha estat en aquella posició
        if sCurrentState not in allMoves.keys():
            # Construim el value de la posició, amb un valor de 0
            allMoves[sCurrentState] = dict()
            allMoves[sCurrentState]['value'] = 0
            allMoves[sCurrentState]['moves'] = dict()
        # Mirem els seus possibles moviments a fer, per tal de construir o actualitzar els values d'aquests
        for move in list_moves:
            # Si el moviment a fer, es dins allMoves, actualitzem el seu value, per l'actual
            if str(move) in allMoves:
                allMoves[sCurrentState]['moves'][str(move)] = allMoves[str(move)]['value']
            # Si no hi es, inicialitzem amb un value de 0
            else:
                allMoves[sCurrentState]['moves'][str(move)] = 0

        return allMoves[sCurrentState]

    def evalState(self, next_state, estatActual, discount_factor, allMoves, alpha):
        """
            - Funció d'avaluar el estat. Anirem actualitzant el value dels estats depenent el proxim moviment. Apliquem TD
            Args:
                next_state: Posició següent, moviment a fer
                estatActual: Poisció del estat on està la peça actualment
                discount_factor: Serveix per limitar que una suma infinita sigui finita. Per demostrar la convergencia
                alpha: Tasa per recordar el value anterior. Si alpha = 1, olvida per complert el value anterior
                allMoves: Diccionari principal, on hi ha tots els estats on ha estat la peça amb els seus values i
                possibles moviments
            Returns: allMoves -> Diccionari principal, amb els values dels estats actualitzat
        """
        reward = 0
        # Si tenim que el proxim moviment es checkMate, tenim una recompensa de 100
        if (self.isCheckMate(next_state)):
            reward = 100
        # Si no, una recompensa de -1
        else:
            reward = -1
        # Apliquem formula TD, i actualitzem el value del estat actual segons el proxim moviment
        td_target = reward + discount_factor * (allMoves[str(next_state)]['value']) - allMoves[str(estatActual)][
            'value']
        allMoves[str(estatActual)]['value'] += td_target * alpha
        return allMoves

    def qLearning(self, currentState, num_episodes, depth, discount_factor=0.4,  alpha=0.6):
        """
            - Funció principal on l'algorisme anirà aprenent i movent les peces.
            Args:
                currentState: Posició actual
                num_episodes: Num de partides que volguem que es jugui desde la posició inicial
                depth: Limit de moviments possibles que fara dintre d'una partida
                discount_factor i alpha, explicat a evalState
            Returns: 0
        """
        # Creem el diccionari, i l'inicialitzem ficant l'estat inicial amb el seu value que serà:
        # el value del estat, i un altre diccionari amb els proxims moviments i el value de cada.
        allMoves = dict()
        proxMovs = self.getListNextStatesW(currentState)
        allMoves[str(currentState)] = self.valueState(currentState, allMoves, proxMovs)
        initState = currentState

        # Iterem tantes vegades com episodis tinguem
        for iteration in range(num_episodes):
            # Reiniciem el tauler per tornar a començar, i tornem a inicialitzar el current state i els seu prox. movmients
            self.moveWhite(initState)
            proxMovs = self.getListNextStatesW(initState)
            currentState = initState
            # Introduim un depth de 1000 perque arribi al checkMate segur. Limit de possibles moviments en una partida
            for j in range(depth):
                # Asignem a next_state el millor possible moviment fet a moveGreedy, i fem el moviment al tauler
                next_state = self.moveGreedy(currentState, proxMovs, allMoves, 0.9)
                self.moveWhite(next_state)
                # Actualitzem el value del moviment cridant a valueState i actualitzem el currentState
                allMoves[str(next_state)] = self.valueState(next_state, allMoves,
                                                            self.getListNextStatesW(next_state))
                allMoves = self.evalState(next_state, currentState, discount_factor, allMoves,alpha)
                currentState = next_state
                # Si arribem al check mate, aquesta partida haurà acabat. Fem un break per tornar a començar
                if(self.isCheckMate(currentState)):
                    #aichess.chess.boardSim.print_board()
                    break

        # Un cop hem entrenat l'algorisme, jugarem la partida final. Fem el mateix procès que abans
        self.moveWhite(initState)
        proxMovs = self.getListNextStatesW(initState)
        currentState = initState
        # Llista on anirem guardant els moviments finals
        finalList = [initState]
        aichess.chess.boardSim.print_board()
        # Ara fiquem un limit de 50 moviments per si de cas, pero entre 5-12 ja finalitzaria.
        for i in range(50):
            # Li donem un epsiol d'1.0, perque sempre faci el millor moviment, cap a l'atzar
            next_state = self.moveGreedy(currentState, proxMovs, allMoves, 1.0)
            self.moveWhite(next_state)
            # Anem imprimint el tauler perque es vegi tota la partida final sencera
            aichess.chess.boardSim.print_board()
            allMoves[str(next_state)] = self.valueState(next_state, allMoves,
                                                        self.getListNextStatesW(next_state))
            allMoves = self.evalState(next_state, currentState, discount_factor, allMoves, alpha)
            currentState = next_state
            # Fiquem el moviment a la llista final
            finalList.append(currentState)
            # Un cop arriba a checkMate, haurem guanyat i imprimint tots els moviments emplarts per guanyar
            if (self.isCheckMate(currentState)):
                print("Final Movmients:", finalList)
                print("Tenim un total de ", len(finalList), "moviments")
                break
        return finalList


    def moveGreedy(self, sCurrentState, listNextStates, allMoves, epsilon):
        """
            - Funció per extreure el millor possible moviment a fer de forma Greedy.
            Hi ha vegades que ho fa de forma aleatoria
            Args:
                scurrentState: Poisció del estat on està la peça actualment
                listNextStates: Proxims possibles moviments a fer desde la posició actual
                allMoves: Diccionari principal, on hi ha tots els estats on ha estat la peça amb els seus values i
                possibles moviments
                epsilon: Constant, l'utilitzem per fer un percentatge de vegades el millor moviment, i unes poques un
                moviment aleatori
            Returns: allMoves[sCurrentState]  -> Value construit o actualitzat del estat actual dins el dicc principal
        """
        # Float aletori entre 0 i 1. Si aquest es més petit que epsilon, escollim el millor moviment possible a fer
        if random.random() < epsilon:
            # Agafem el possible moviment amb millor Value
            strs = max(allMoves[str(sCurrentState)]['moves'], key=allMoves[str(sCurrentState)]['moves'].get)
            sNext_State = json.loads(strs)
            # Controlem que no es solapi amb l'altre peça
            if not self.overLayer(sNext_State):
                return sNext_State
            # Si es solapa farem un next moviment aleatori
            else:
                aleatori = np.random.choice(len(listNextStates), 1)
                return listNextStates[int(aleatori)]
        # Un petit percentatge de vegades fem un moviment aleatori perque busqui altres camins inexplorats
        else:
            aleatori = np.random.choice(len(listNextStates), 1)
            return listNextStates[int(aleatori)]

    def minMaxQLearning(self, currentStateW, currentStateB, maximizingPlayer):
        # Inicialitzem el moviment inicial, construint el seu value
        allMoves = dict()
        proxMovs = self.getListNextStatesW(currentStateW)
        allMoves[str(currentStateW)] = self.valueState(currentStateW, allMoves, proxMovs)

        # Si li toca jugar al maximizingPlayer
        if (maximizingPlayer):
            # Fem el qLearning amb les peces blanques
            nextListW = self.qLearning(currentStateW, 30, 100)
            # Treiem el primer moviment que serà l'estat incial i més endevant el moviment actual
            nextListW.pop(0)
            # Cridem recursivament al minMaxQLearning pero la posicio de les blanques es el primer moviment de la llista
            aichess.minMaxQLearning(nextListW[0], currentStateB, False)

        # Si li toca al minimingPlayer
        else:
            # Fem el mateix que abans pero ara amb les negres
            nextListB = self.qLearning(currentState, 30, 100)
            nextListB.pop(0)
            aichess.minMaxQLearning(currentStateW, nextListB[0], True)


"""
def BestFirstSearch(self, currentState):





def AStarSearch(self, currentState):


      """


def translate(s):
    """
    Translates traditional board coordinates of chess into list indices
    """

    try:
        row = int(s[0])
        col = s[1]
        if row < 1 or row > 8:
            print(s[0] + "is not in the range from 1 - 8")
            return None
        if col < 'a' or col > 'h':
            print(s[1] + "is not in the range from a - h")
            return None
        dict = {'a': 0, 'b': 1, 'c': 2, 'd': 3, 'e': 4, 'f': 5, 'g': 6, 'h': 7}
        return (8 - row, dict[col])
    except:
        print(s + "is not in the format '[number][letter]'")
        return None


if __name__ == "__main__":
    #   if len(sys.argv) < 2:
    #       sys.exit(usage())

    # intiialize board
    TA = np.zeros((8, 8))
    # white pieces
    # TA[0][0] = 2
    # TA[2][4] = 6
    # # black pieces
    # TA[0][4] = 12
    TA[7][7] = 2
    TA[7][3] = 6
    TA[0][4] = 12
    #TA[0][0] = 8
    # initialise board
    print("stating AI chess... ")
    aichess = Aichess(TA, True)
    currentState = aichess.chess.board.currentState.copy()
    currentStateB = aichess.chess.board.currentStateB.copy()
    currentStateW = aichess.chess.board.currentStateW.copy()

    print("printing board")
    aichess.chess.boardSim.print_board()

    # get list of next states for current state
    # print("current State ", currentState)
    # print("current State Blacks", currentStateB)
    # print("current State Whites", currentStateW)

    # it uses board to get them... careful

    #   aichess.getListNextStatesW([[7,4,2],[7,4,6]])
    #print("list next states ", aichess.listNextStates)

    # starting from current state find the end state (check mate) - recursive function
    # aichess.chess.boardSim.listVisitedStates = []
    # find the shortest path, initial depth 0
    depth = 0
    # aichess.BreadthFirstSearch(currentState)
    # aichess.DepthFirstSearch(currentState, depth)
    # aichess.torre(currentStateW, currentStateB)
    # moviment = dict()
    # aichess.minimax(currentStateW, currentStateB, 1, True, dict())

    aichess.qLearning(currentStateW, 300, 1000)
    # MovesToMake = ['1e','2e','2e','3e','3e','4d','4d','3c']

    # for k in range(int(len(MovesToMake)/2)):

    #     print("k: ",k)

    #     print("start: ",MovesToMake[2*k])
    #     print("to: ",MovesToMake[2*k+1])

    #     start = translate(MovesToMake[2*k])
    #     to = translate(MovesToMake[2*k+1])

    #     print("start: ",start)
    #     print("to: ",to)

    #     aichess.chess.moveSim(start, to)

    # aichess.chess.boardSim.print_board()
    #print("#Move sequence...  ", aichess.pathToTarget)
    #print("#Visited sequence...  ", aichess.listVisitedStates)
    #print("#Current State...  ", aichess.chess.board.currentStateW)
    #print("#Depth...  ", aichess.finaldepth)
    # print("list next states 2 ", aichess.listNextStates)
