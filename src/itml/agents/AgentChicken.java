package itml.agents;

import itml.cards.Card;
import itml.cards.CardRest;
import itml.simulator.CardDeck;
import itml.simulator.StateAgent;
import itml.simulator.StateBattle;
import weka.classifiers.Classifier;
import weka.core.Instances;

import java.util.ArrayList;

/**
 *
 *  Example agent:
 *
 * This agent always tries to run away from the other agent (chicken, chicken, ...).
 * If it cannot do so, it rests to gain stamina.
 *
 * @author      Yngvi Bjornsson
 *
 * @version     %I%, %G%
 *
 */
public class AgentChicken extends Agent {

    private int m_noThisAgent;     // Index of our agent (0 or 1).
    private int m_noOpponentAgent; // Inex of opponent's agent.

    public AgentChicken( CardDeck deck, int msConstruct, int msPerMove, int msLearn ) {
        super(deck, msConstruct, msPerMove, msLearn );
    }

    public void startGame(int noThisAgent, StateBattle stateBattle) {
        // Remember the indicies of the agents in the StateBattle.
        m_noThisAgent = noThisAgent;
        m_noOpponentAgent  = (noThisAgent == 0 ) ? 1 : 0; // can assume only 2 agents battling.
    }

    public void endGame(StateBattle stateBattle, double[] results) {
        // Nothing to do.
    }

    public Card act(StateBattle stateBattle ) {

        Card [] move = new Card[2];

        move[m_noOpponentAgent] = new CardRest();   // We assume the opponent just stays where he/she is,
                                                    // and then take the move that brings us as far away as possible.

        Card bestCard = new CardRest();
        int  minDistance = calcDistanceBetweenAgents( stateBattle );

        ArrayList<Card> cards = m_deck.getCards( stateBattle.getAgentState( m_noThisAgent ).getStaminaPoints() );
        for ( Card card : cards ) {
            StateBattle bs = (StateBattle) stateBattle.clone();   // close the state, as play( ) modifies it.
            move[m_noThisAgent] = card;
            bs.play( move );
            int  distance = calcDistanceBetweenAgents( bs );
            if ( distance > minDistance ) {
                bestCard = card;
                minDistance = distance;
            }
        }

        return bestCard;
    }

    public Classifier learn( Instances instances ) {
        // no learning
        return null;
    }

    private int calcDistanceBetweenAgents( StateBattle bs ) {

        StateAgent asFirst = bs.getAgentState( 0 );
        StateAgent asSecond = bs.getAgentState( 1 );

        return Math.abs( asFirst.getCol() - asSecond.getCol() ) + Math.abs( asFirst.getRow() - asSecond.getRow() );
    }

}
