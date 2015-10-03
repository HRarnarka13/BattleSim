package itml.agents;

import itml.cards.Card;
import itml.simulator.CardDeck;
import itml.simulator.StateBattle;
import weka.classifiers.Classifier;
import weka.core.Instances;

/**
 * Created by arnarkari on 03/10/15.
 *
 * @author arnarkari
 */
public class AgentFresco extends Agent {

    public AgentFresco(CardDeck deck, int msConstruct, int msPerMove, int msLearn) {
        super(deck, msConstruct, msPerMove, msLearn);
    }

    @Override
    public void startGame(int noThisAgent, StateBattle stateBattle) {

    }

    @Override
    public void endGame(StateBattle stateBattle, double[] results) {

    }

    @Override
    public Card act(StateBattle stateBattle) {
        return null;
    }

    @Override
    public Classifier learn(Instances instances) {
        return null;
    }
}
