package itml.agents;

import itml.cards.Card;
import itml.cards.CardRest;
import itml.simulator.CardDeck;
import itml.simulator.StateBattle;
import weka.classifiers.Classifier;
import weka.core.Instances;

/**
 *
 *  Example agent:
 *
 * This agent is simply a sitting duck.
 *
 * @author      Yngvi Bjornsson
 *
 * @version     %I%, %G%
 *
 */
public class AgentLazy extends Agent {

    public AgentLazy( CardDeck deck, int msConstruct, int msPerMove, int msLearn ) {
        super(deck, msConstruct, msPerMove, msLearn );
    }

    public void startGame(int noThisAgent, StateBattle stateBattle) {
        // No book-keeping needed.
    }

    public void endGame( StateBattle stateBattle, double [] results ) {
        // No book-keeping needed.
    }

    public Card act( StateBattle stateBattle ) {
        return new CardRest();
    }

    public Classifier learn( Instances instances  ) {
        // Too lazy to learn anything.
        return null;
    }
}
