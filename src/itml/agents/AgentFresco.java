package itml.agents;

import itml.cards.Card;
import itml.cards.CardDefend;
import itml.cards.CardRest;
import itml.simulator.CardDeck;
import itml.simulator.StateAgent;
import itml.simulator.StateBattle;
import weka.classifiers.Classifier;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.trees.J48;
import weka.core.Instance;
import weka.core.Instances;

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
    private final int GRID_SIZE = 4;

    private enum Direction {RIGHT, LEFT, UP, DOWN}

    /**
     * Calculates the distance between two agents
     * @param sb current state of battle
     * @return the distance between agents
     */
    private int distanceBetweenAgents(StateBattle sb) {
        StateAgent asFirst = sb.getAgentState( 0 );
        StateAgent asSecond = sb.getAgentState( 1 );

        return Math.abs( asFirst.getCol() - asSecond.getCol() ) + Math.abs( asFirst.getRow() - asSecond.getRow() );
    }

    /**
     * Finds the card that brings us closest to the opponent
     * @param availableCards list of currently available cards
     * @param sb current state of battle
     * @return the best card
     */
    private Card minimizeDistanceCard(List<Card> availableCards, StateBattle sb) {
        Card bestCard = new CardRest();
        int bestDistance = distanceBetweenAgents(sb);
        Card [] move = new Card[2];
        move[m_noOpponentAgent] = new CardRest();
        for (Card card : availableCards) {
            move[m_noThisAgent] = card;
            sb.play(move);
            int  distance = distanceBetweenAgents(sb);
            if (distance < bestDistance) {
                bestCard = card;
                bestDistance = distance;
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
    public AgentFresco( CardDeck deck, int msConstruct, int msPerMove, int msLearn ) {
        super(deck, msConstruct, msPerMove, msLearn);
        classifier_ = new J48();
//        classifier_ = new NaiveBayes();
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

    @Override
    public Card act(StateBattle stateBattle) {
        StateBattle bs = (StateBattle) stateBattle.clone();   // close the state, as play( ) modifies it.
        double[] values = new double[8];
        StateAgent a = stateBattle.getAgentState(0);
        StateAgent o = stateBattle.getAgentState(1);
        values[0] = a.getCol();
        values[1] = a.getRow();
        values[2] = a.getHealthPoints();
        values[3] = a.getStaminaPoints();
        values[4] = o.getCol();
        values[5] = o.getRow();
        values[6] = o.getHealthPoints();
        values[7] = o.getStaminaPoints();
        try {
            ArrayList<Card> allCards = m_deck.getCards(); // all cards
            ArrayList<Card> cards = m_deck.getCards(a.getStaminaPoints());// cards that we have stamina to use
            Instance i = new Instance(1.0, values.clone());
            i.setDataset(dataset);
            int out = (int)classifier_.classifyInstance(i);
            Card selected = allCards.get(out);
            System.out.println("Our  guess = " + selected.getName());

            // What to do if the opponent is likely to attack
            if(selected.getType().equals(Card.CardActionType.ctAttack)){
                Card defend = new CardDefend();
                for(Card c : cards){
                    if (c.getType().equals(Card.CardActionType.ctDefend)) {
                        // find the best defence card if there are many?
                        System.out.println(c.getName());
                        return c;
                    }
                }
            }

            if(cards.contains(selected)) {
                System.out.println(selected.getName());
                return selected;
            }
        } catch (Exception e) {
            System.out.println("Error classifying new instance: " + e.toString());
        }
        return new CardRest();  //To change body of implemented methods use File | Settings | File Templates.
    }

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
