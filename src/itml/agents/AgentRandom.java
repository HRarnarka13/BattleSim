package itml.agents;

import itml.cards.Card;
import itml.simulator.CardDeck;
import itml.simulator.StateAgent;
import itml.simulator.StateBattle;
import weka.classifiers.Classifier;
import weka.core.Instances;
import java.util.ArrayList;
import java.util.Random;

/**
 *
 *  Example agent:
 *
 * This agent picks a random legal action.
 *
 * @author      Yngvi Bjornsson
 *
 * @version     %I%, %G%
 *
 */
public class AgentRandom extends Agent {

    private int m_noThisAgent;
    
    static private Random m_random = new Random();  // A random number generator, for randomly picking actions.

    public AgentRandom( CardDeck deck, int msConstruct, int msPerMove, int msLearn ) {
        super(deck, msConstruct, msPerMove, msLearn );
    }

    public void startGame(int noThisAgent, StateBattle stateBattle) {
        m_noThisAgent = noThisAgent;
    }

    public void endGame(StateBattle stateBattle, double [] results ) {
        // No book-keeping needed.
    }

    public Card act(StateBattle stateBattle) {

        StateAgent stateAgent = stateBattle.getAgentState( m_noThisAgent );
        
        ArrayList<Card> cards = m_deck.getCards( stateAgent.getStaminaPoints() );

        return cards.get( m_random.nextInt( cards.size() ) );
    }

    public Classifier learn( Instances instances ) {
        // No learning.
        return null;
    }

}
