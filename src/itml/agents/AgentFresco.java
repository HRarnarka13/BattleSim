package itml.agents;

import itml.cards.Card;
import itml.cards.CardRest;
import itml.simulator.CardDeck;
import itml.simulator.GameLog;
import itml.simulator.StateAgent;
import itml.simulator.StateBattle;
import weka.classifiers.Classifier;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.trees.J48;
import weka.core.Instance;
import weka.core.Instances;

import java.lang.reflect.Array;
import java.util.ArrayList;
import java.util.List;

/**
 * User: deong
 * Date: 9/28/14
 */
public class AgentFresco extends Agent {
    private int m_noThisAgent;     // Index of our agent (0 or 1).
    private int m_noOpponentAgent; // Inex of opponent's agent.
    private Classifier classifier_;
    private Instances dataset;
    private Card lastPredict;
    private Card ourLastMove;

    public int totalCorrect = 0;
    public int totalWrong = 0;
    public int totalSteps = 0;
    public int totalWrongType = 0;
    public int totalCorrectType = 0;


    private final int GRID_SIZE = 4;
    private final int MAXIMUM_STAMINA = 10;
    private enum Direction {RIGHT, LEFT, UP, DOWN}

    // region Helpers


    /**
     * This function returns how many tiles a player can move in a given direction without
     * going out of the grid.
     * @param col the current column of the player
     * @param row the current row of the player
     * @param direction the direction we want to move to.
     * @return How many tiles can we move over without going out of the grid
     */
    private int distanceFromEdge(int col, int row, Direction direction) {
        switch (direction) {
            case RIGHT:
                return GRID_SIZE - col;
            case LEFT:
                return Math.abs(0 - col);
            case UP:
                return GRID_SIZE - row;
            case DOWN:
                return Math.abs(0 - row);
            default:
                throw new IllegalArgumentException("Illegal direction.");
        }
    }

    /**
     * This function decides whether a player can play a particular card or not,
     * depending on his stamina and the staminaPoints of the card.
     * @param stamina The current stamina of the player in question.
     * @param card The card the player wants to play.
     * @return If the player has got enough stamina to play the card.
     */
    private boolean enoughStamina(int stamina, Card card) {
        return (card.getStaminaPoints() > stamina);
    }

    /**
     * Calculates the distance between two agents
     * @param sb current state of battle
     * @return the distance between agents
     */
    private int distanceBetweenAgents(StateBattle sb) {
        StateAgent asFirst = sb.getAgentState( 0 );
        StateAgent asSecond = sb.getAgentState(1);

        return Math.abs( asFirst.getCol() - asSecond.getCol() ) + Math.abs( asFirst.getRow() - asSecond.getRow() );
    }

    /**
     * Check if we are on the grid in current state
     * @param sb state of the battle now
     * @return true if we are out of bounds, false if we are still on the field
     */
    private boolean outOfBounds(StateBattle sb) {
        StateAgent a = sb.getAgentState(m_noThisAgent);
        int column = a.getCol();
        int row = a.getRow();
        if(column > 4 || row > 4 || column < 0 || row < 0) {
            return true;
        } else {
            return false;
        }
    }

    /**
     * Finds the card that brings us closest to the opponent
     * @param availableCards list of currently available cards
     * @param sb current state of battle
     * @return the best card
     */
    private Card minimizeDistanceCard(ArrayList<Card> availableCards, StateBattle sb, Card predictedCard) {
        ArrayList<Card> safeZoneCards = new ArrayList<Card>();
        StateAgent a = sb.getAgentState(m_noThisAgent);
        Card [] move = new Card[2];
        move[m_noOpponentAgent] = predictedCard;
        int currentHealthPoints = a.getHealthPoints();
        for (Card card : availableCards) {
            StateBattle bs = (StateBattle) sb.clone();   // close the state, as play( ) modifies it.
            move[m_noThisAgent] = card;
            bs.play(move);
            // if this move  does not reduce our healthpoints we add to the list
            int healthPointAfterMove = bs.getAgentState(m_noThisAgent).getHealthPoints();

            if (currentHealthPoints == healthPointAfterMove && !outOfBounds(bs)) {
               safeZoneCards.add(card);
            }
        }
        Card bestCard = safeZoneCards.get(0);
        int bestDistance = distanceBetweenAgents(sb);
        for(Card c : safeZoneCards){
            move[m_noThisAgent] = c;
            StateBattle bs2 = (StateBattle) sb.clone();   // close the state, as play( ) modifies it.
            bs2.play(move);
            if(distanceBetweenAgents(bs2) < bestDistance){
                bestDistance = distanceBetweenAgents(bs2);
                bestCard = c;
            }

        }
        return bestCard;
    }

    /**
     * Returns true if both agents are on the same field.
     * @param a our agent
     * @param o opponent agent
     * @return True if on same field.
     */
    private boolean agentsOnSameSquare(StateAgent a, StateAgent o) {
        return a.getCol() == o.getCol() && a.getRow() == o.getRow();
    }

    /**
     * Calculate which attack card to use. First add all cards that will hit
     * to a list then pick the highest damaging one.
     * @param cards list of attack cards
     * @param a our agent
     * @param o opponent
     * @param sb statbattle
     * @param predictedCard the predict opponent card
     * @return the best card to attack with
     */
    private Card whichAttackToUse(ArrayList<Card> cards, StateAgent a, StateAgent o, StateBattle sb, Card predictedCard){

        Card restCard = new CardRest();
        Card [] move = new Card[2];
        move[m_noOpponentAgent] = predictedCard;
        ArrayList<Card> cardsThatHit = new ArrayList<Card>();
        int currentOHealthpoints = o.getHealthPoints();
        for(Card c : cards){
//            System.out.println("Attack card " + c.getName() + " Stamina required " + c.getStaminaPoints() + " Our stamina " +  a.getStaminaPoints());
            StateBattle bs = (StateBattle) sb.clone();   // close the state, as play( ) modifies it.
            move[m_noThisAgent] = c;
            bs.play(move);
            StateAgent opponentAfterMove = bs.getAgentState(m_noOpponentAgent);
            // if attack will hit add it to the list
            if(c.inAttackRange(bs.getAgentState(m_noThisAgent).getCol(),
                               bs.getAgentState(m_noThisAgent).getRow(),
                               bs.getAgentState(m_noOpponentAgent).getCol(),
                               bs.getAgentState(m_noOpponentAgent).getRow())
                               && currentOHealthpoints > bs.getAgentState(m_noOpponentAgent).getHealthPoints() ){
                cardsThatHit.add(c);
            }
        }
//        System.out.println("how many cards that hit " + cardsThatHit.size());
        // if we dont find any card, return the rest card
        if(cardsThatHit.isEmpty()){
            return restCard;
        }
        Card bestCard = cardsThatHit.get(0);
        // we already checked if we have enough stamina to use the card, so we just pick the highest damaging one
        for(Card c : cardsThatHit){
            if(c.getHitPoints() > bestCard.getHitPoints()){
                bestCard = c;
            }
        }
//        System.out.println("Best attack card  = " + bestCard.getName());
        return bestCard;
    }

    /**
     *
     * @param selected
     * @param sb
     * @return
     */
    private boolean opponentAttackWillHit(Card selected, StateBattle sb) {
        StateBattle bs = (StateBattle) sb.clone();
        Card ourMove = new CardRest();
        Card [] move = new Card[2];
        move[m_noThisAgent] = ourMove;
        move[m_noOpponentAgent] = selected;
        int aCurrHealthPoints = bs.getAgentState(m_noThisAgent).getHealthPoints();
        bs.play(move);
        if(aCurrHealthPoints > bs.getAgentState(m_noThisAgent).getHealthPoints()) {
            return true;
        } else {
            return false;
        }
    }
    // endregion

    public AgentFresco( CardDeck deck, int msConstruct, int msPerMove, int msLearn ) {
        super(deck, msConstruct, msPerMove, msLearn);
//        classifier_ = new J48();
//        classifier_ = new NaiveBayes();
        classifier_ = new J48();
    }

    @Override
    public void startGame(int noThisAgent, StateBattle stateBattle) {
        // Remember the indicies of the agents in the StateBattle.
        m_noThisAgent = noThisAgent;
        m_noOpponentAgent  = (noThisAgent == 0 ) ? 1 : 0; // can assume only 2 agents battling.
    }

    @Override
    public void endGame(StateBattle stateBattle, double[] results) {
        //To change body of implemented methods use File | Settings | File Templates.
    }
    public Card predictCard(double[] values, ArrayList<Card> allCards) throws Exception {
        Instance i = new Instance(1.0, values.clone());
        i.setDataset(dataset);
        int out = (int)classifier_.classifyInstance(i);
        Card selected = allCards.get(out);
        return selected;
    }

    @Override
    public Card act(StateBattle stateBattle) {

        System.out.println();
        System.out.println("************" + totalSteps + "*************");
        System.out.println("Overall \t Total type correct");
        System.out.println("" + totalCorrect + "\t" + totalSteps + "\t" + totalCorrectType + "\t" + totalSteps);
        if (ourLastMove != null) {
            System.out.println("ourLastMove.getName() = " + ourLastMove.getName());
        }

        StateBattle sb = (StateBattle) stateBattle.clone();   // close the state, as play( ) modifies it.
        System.out.println();
        double[] values = new double[8];
        StateAgent a = stateBattle.getAgentState(m_noThisAgent);
        StateAgent o = stateBattle.getAgentState(m_noOpponentAgent);

        boolean foundOpponentCard = false;
        Card opponentCard = null;
        for (Card c : sb.getLastMoves() ) {
            if (c != null && ourLastMove != null && ourLastMove.getName() != c.getName()) {
                foundOpponentCard = true;
                opponentCard = c;
            }
        }
        if (ourLastMove !=null && foundOpponentCard == false) {
            opponentCard = ourLastMove;
        }

        if (opponentCard != null) {
            System.out.println("Enemy last move = " + opponentCard.getName());
        }



        if (opponentCard != null && lastPredict != null ) {
            if (opponentCard.getType().equals(lastPredict.getType())) {
                totalCorrectType++;
            } else {
                totalWrongType++;
            }

            if (opponentCard.getName().equals(lastPredict.getName())) {
                totalCorrect++;
            } else {
                totalWrong++;
            }
            System.out.println("OpponentCard: " + opponentCard.getName() + " " + opponentCard.getType().name() +
                            " LastPredict: " + lastPredict.getName() + " " + lastPredict.getType().name());
        }
        totalSteps++;


        values[0] = a.getCol();
        values[1] = a.getRow();
        values[2] = a.getHealthPoints();
        values[3] = a.getStaminaPoints();
        values[4] = o.getCol();
        values[5] = o.getRow();
        values[6] = o.getHealthPoints();
        values[7] = o.getStaminaPoints();
        System.out.println("AgentFresco : " + m_noThisAgent + " Looser : " + m_noOpponentAgent);
        try {
            ArrayList<Card> allCards = m_deck.getCards(); // all cards
            ArrayList<Card> cards = m_deck.getCards(a.getStaminaPoints());// cards that we have stamina to use

            ArrayList<Card> attackCards = new ArrayList<>();
            ArrayList<Card> defendCards = new ArrayList<>();
            ArrayList<Card> moveCards = new ArrayList<>();
            for ( Card c : cards ) {
                if (c.getType().equals(Card.CardActionType.ctAttack)) {
                    attackCards.add(c);
                } else if (c.getType().equals(Card.CardActionType.ctDefend)) {
                    defendCards.add(c);
                } else if (c.getType().equals(Card.CardActionType.ctMove)) {
                    moveCards.add(c);
                } else {
                    throw new Exception("Unknown type of card");
                }
            }
            Card returnCard;
            Card selected = predictCard(values, allCards);
            lastPredict = selected;
            String ourGuess = selected.getName();
            // if the opponent does not have any stamina we attack him no matter what
            if(o.getStaminaPoints() < 1){
                returnCard = whichAttackToUse(attackCards, a, o, sb, new CardRest());
                ourLastMove = returnCard;
                return returnCard;
            }

            // What to do if opponent attacks
            Card.CardActionType cardType = selected.getType();
            if (cardType.equals(Card.CardActionType.ctAttack)) {// Opponent about to attack
                if(opponentAttackWillHit(selected, sb)) {
                    // if we are stronger, attack
                    if (a.getStaminaPoints() > o.getStaminaPoints() && a.getHealthPoints() > o.getHealthPoints()) {
//                        System.out.println("Attack because we have more HP");
                        returnCard = whichAttackToUse(attackCards, a, o, sb, selected);
                        ourLastMove = returnCard;
                        return returnCard;
                    } else {
//                        System.out.println("Dodge dip duck dive and dodge");
//                        System.out.println(minimizeDistanceCard(moveCards, sb, selected).getName());
                        returnCard = minimizeDistanceCard(moveCards, sb, selected); // DANCE, dodge the attack
                        ourLastMove =  returnCard;
                        return returnCard;
                    }
                } else {
//                    System.out.println("Opponent missing his attack, attack him ");
//                    System.out.println(whichAttackToUse(attackCards, a, o, sb, selected).getName());
                    if(whichAttackToUse(attackCards, a, o, sb, selected).getName().equals("cRest")){
//                        System.out.println("if attack to use != rest");
                        returnCard = minimizeDistanceCard(moveCards, sb, selected);
                        ourLastMove = returnCard;
                        return returnCard;
                    }
                    returnCard = whichAttackToUse(attackCards, a, o, sb, selected);
                    ourLastMove = returnCard;
                    return returnCard;
                }
                // if opponent is defending
            } else if (cardType.equals(Card.CardActionType.ctDefend)) { // Opponent about to defend
                if (a.getStaminaPoints() + new CardRest().getStaminaPoints() <= MAXIMUM_STAMINA ) {
                    returnCard = new CardRest(); // if the agent benefits from resting, the agent rests
                    ourLastMove = returnCard;
                    return returnCard;
                } else if (selected.inAttackRange(a.getCol(), a.getRow(), o.getCol(), o.getRow())) {
                    returnCard = whichAttackToUse(attackCards, a, o, sb, selected);
                    ourLastMove = returnCard;
                    return returnCard;
                } else { // Move closer to the opponent
                    returnCard = minimizeDistanceCard(cards, sb, selected); // return the best move card
                    ourLastMove = returnCard;
                    return returnCard;
                }
            //if opponent is moving
            } else if (cardType.equals(Card.CardActionType.ctMove)) { // Opponent about to move
                if(selected.getName().equals("cRest")){
//                    System.out.println("tessi if setning gaeti verid vitlaus");
                    if(whichAttackToUse(attackCards, a, o, sb, selected).getName().equals("cRest")){
//                        System.out.println("er tad ad skila okkur resT???" + minimizeDistanceCard(moveCards, sb, selected).getName());
                        returnCard = minimizeDistanceCard(moveCards, sb, selected);
                        ourLastMove = returnCard;
                        return returnCard;
                    } else{
                        returnCard = whichAttackToUse(attackCards, a, o, sb, selected);
                        ourLastMove = returnCard;
                        return returnCard;
                    }

                } else {
                    //
                    if(whichAttackToUse(attackCards, a, o, sb, selected).getName().equals("cRest")){
                        returnCard = minimizeDistanceCard(moveCards, sb, selected);
                        ourLastMove = returnCard;
                        return returnCard;
                    }
                    returnCard = whichAttackToUse(attackCards, a, o, sb, selected);
                    ourLastMove = returnCard;
                    return returnCard;
                }
            }
        } catch (Exception e) {
            System.out.println("Error classifying new instance: " + e.toString());
        }
        return new CardRest();  //To change body of implemented methods use File | Settings | File Templates.
    }

    @Override
    public Classifier learn(Instances instances) {
        this.dataset = instances;
        try {
            classifier_.buildClassifier(instances);
        } catch(Exception e) {
            System.out.println("Error training classifier: " + e.toString());
        }
        return null;  //To change body of implemented methods use File | Settings | File Templates.
    }
}
