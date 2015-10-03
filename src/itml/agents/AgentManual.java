package itml.agents;

import itml.cards.Card;
import itml.cards.CardRest;
import itml.simulator.CardDeck;
import itml.simulator.StateAgent;
import itml.simulator.StateBattle;
import weka.classifiers.Classifier;
import weka.core.Instances;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.ArrayList;

/**
 *
 *  Example agent:
 *
 * This agent allows a user to play as an agent (prompted for actions).
 *
 * @author      Yngvi Bjornsson
 *
 * @version     %I%, %G%
 *
 */
public class AgentManual extends Agent {


    private int m_noThisAgent;   // Index of this agent.

    public AgentManual( CardDeck deck, int msConstruct, int msPerMove, int msLearn ) {
        super(deck, msConstruct, msPerMove, msLearn );
    }

    public void startGame(int noThisAgent, StateBattle stateBattle) {
        m_noThisAgent = noThisAgent;
    }

    public void endGame(StateBattle stateBattle, double[] results) {
        // No book-keeping needed.
    }

    public Card act(StateBattle stateBattle) {

        StateAgent as = stateBattle.getAgentState(m_noThisAgent);

        ArrayList<Card> cards = m_deck.getCards( as.getStaminaPoints() );

        System.out.println( stateBattle.toString() );
        System.out.print( "Choose an action ('q' for quit):" );
        for ( Card card : cards ) {
            System.out.print( ' ' + card.getName()  );
        }
        System.out.println();

        try {
            BufferedReader in = new BufferedReader(new InputStreamReader(System.in));
            String str = "";
            while (str != null) {
                System.out.print("> ");
                str = in.readLine();
                if ( str.equals("q") ) {
                    return null;  // exit.
                }
                for ( Card card : cards ) {
                    if ( card.getName().equals( str ) ) {
                        return card;
                    }
                }
                System.out.println( "Sorry, illegal choice.");
            }
        } catch (IOException e) {
        }

        return new CardRest();
    }


    public Classifier learn( Instances instances  ) {
        // This agent does not do any learning.
        return null;
    }

}
