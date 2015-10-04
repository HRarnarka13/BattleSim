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

/**
 * User: deong
 * Date: 9/28/14
 */
public class AgentFresco extends Agent {
    private int m_noThisAgent;     // Index of our agent (0 or 1).
    private int m_noOpponentAgent; // Inex of opponent's agent.
    private Classifier classifier_;
    private Instances dataset;

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
