package itml;

import itml.agents.*;
import weka.core.Attribute;
import weka.core.FastVector;
import weka.core.Instance;
import weka.core.Instances;
import java.io.BufferedWriter;
import java.io.FileWriter;
import java.util.Random;
import itml.simulator.*;
import itml.cards.*;


/**
 *
 *  The main class, drives the training and competition matches.
 *
 * @author      Yngvi Bjornsson
 *
 * @version     %I%, %G%
 *
 */
public class BattleSim {

    /**
     *  This is the main routine.
     *
     * @param  args  Command line arguments
     *
     *      itml.BattleSim  [ <numSteps> <numTrainingGames> <numPlayingGames> </numPlayingGames><msConstuctor> <msPerMove> <msLearning> ]
     */
	public static void main(String [] args)
	{
        System.out.println( "Welcome to BattleSim 2.1.3" );

        // Default arguments.
        int numStepsInGame   = 30;     // Maximum step length of a game.
        int numTrainingGames = 10;     // Number of games to play in the training phase.
        int numPlayingGames  = 100;    // Number of games to play in the evaluation phase.
        int msConstruct      = 5000;   // Maximum time to use in Agent constructor (in ms.)
        int msPerMove        = 50;     // Maximum time to use per act, startGame, endGame call.
        int msLearning       = 30000;  // Maximum time to use in the learning call.

        // Check if any command line arguments
        if (args.length > 0) {
            if ( args.length == 6 ) {
                try {
                    numStepsInGame   = Integer.parseInt(args[0]);
                    numTrainingGames = Integer.parseInt(args[1]);
                    numPlayingGames  = Integer.parseInt(args[2]);
                    msConstruct      = Integer.parseInt(args[3]);
                    msPerMove        = Integer.parseInt(args[4]);
                    msLearning       = Integer.parseInt(args[5]);
                } catch (NumberFormatException e) {
                    System.err.println("Argument must be an integer");
                    System.exit(1);
                }
            }
        }
        
        // Set up the deck of cards.
        CardDeck deck = new CardDeck();
        deck.addCard( new CardRest() );
        deck.addCard( new CardMoveUp() );
        deck.addCard( new CardMoveDown() );
        deck.addCard( new CardMoveLeft() );
        deck.addCard( new CardMoveRight() );
        deck.addCard( new CardLeapLeft() );
        deck.addCard( new CardLeapRight() );
        deck.addCard( new CardDefend() );
        deck.addCard( new CardAttackCardinal() );
        deck.addCard( new CardAttackDiagonal() );
        deck.addCard( new CardAttackLong() );

        // Set up the initial state of the agent (location, health- and stamina-points),
        // and create a battle arena (5 x 5).
        StateAgent[] stateAgents = new StateAgent[2];
        stateAgents[0] =  new StateAgent( 1, 2, 10, 3 );
        stateAgents[1] =  new StateAgent( 3, 2, 10, 3 );
        Battle battle = new Battle( 5, 5, deck, stateAgents);

        // Create agents that will compete.
        long  msStart, msDuration;

        msStart = System.currentTimeMillis();
        Agent agentMy = new AgentRandom( deck.clone(), msConstruct, msPerMove, msLearning );   // The first agent is yours -- change to yours.
        msDuration = System.currentTimeMillis() - msStart;
        System.out.println("Timing agent constructor = " + msDuration );
        if ( msDuration > msConstruct ) {
           System.out.println("WARNING: exceeded time limit (" + msDuration + ">" + msConstruct + ")");
        }

        msStart = System.currentTimeMillis();
        Agent agentOpp = new AgentTerminator( deck.clone(), msConstruct, msPerMove, msLearning );   // The second agent is your opponent.
        msDuration = System.currentTimeMillis() - msStart;
        System.out.println("Timing agent constructor = " + msDuration );
        if ( msDuration > msConstruct ) {
           System.out.println("WARNING: exceeded time limit (" + msDuration + ">" + msConstruct + ")");
        }

        // Now generate the training data for you to observe to predict your opponent's actions;
        // for that we have the opponent play multiple matches against various sparring partners.
        Agent[] agentsSparringPartners = {
                  new AgentChicken( deck.clone(), msConstruct, msPerMove, msLearning ),
                  new AgentRandom( deck.clone(), msConstruct, msPerMove, msLearning ),
                  new AgentTerminator( deck.clone(), msConstruct, msPerMove, msLearning )
               };
        Instances instances = generateTrainingData( battle, numTrainingGames, numStepsInGame, msPerMove,
                                                    agentOpp, agentsSparringPartners );
        try {
            BufferedWriter writer = new BufferedWriter(new FileWriter( "history.arff"));
            writer.write( instances.toString() );
            writer.close();
        }
        catch ( Exception e ) {
            System.err.println( "Warning: could not write out ARFF file");
        }

        // Give your agent the opportunity to learn.
        System.out.println( "Learning ..." );
        msStart = System.currentTimeMillis();
        agentMy.learn(instances);
        msDuration = System.currentTimeMillis() - msStart;
        System.out.println("Timing agent learner = " + msDuration);
        if (msDuration > msLearning) {
            System.out.println("WARNING: exceeded time limit (" + msDuration + ">" + msLearning + ")");
        }

        // Run match games (alternate agent order), and keep track of the score.
        System.out.println( "Match games: " + numPlayingGames );
        Agent[] agents = new Agent[2];
        double [] score = new double[2];
        double scoreMy = 0.0;
        double scoreOpp = 0.0;
        GameLog log = new GameLog();
        for ( int n=0; n < numPlayingGames ; n++ ) {
            int  indexMyAgent  = n % 2;
            int  indexOppAgent = (indexMyAgent == 0) ? 1 : 0;
            agents[indexMyAgent] = agentMy;
            agents[indexOppAgent] = agentOpp;
            battle.run( false, numStepsInGame, msPerMove, agents, score, log );
            scoreMy += score[indexMyAgent];
            scoreOpp += score[indexOppAgent];
        }
        System.out.println( "My score = " + scoreMy + "  Opponent score = " + scoreOpp );
        System.out.println();
    }

    /**
     *  This function creates the Instances object with all the training/test data, by matching
     *  <code>agent</code> against its sparring partners <code>agentsSparring</code>.
     *
     * @param  battle            An battle object, specifying the arena setup.
     * @param  numTrainingGames  An integer representing the number of training games to run.
     * @param  numStepsInGame    An integer representing the maximum number of steps (turns) in a game.
     * @param  msPerMove         An integer representing the maximum time (in milliseconds) an agent can take for an action.
     * @param  agent             The agent that will be matched against the different sparring partners.
     * @param  agentsSparring    An array with the sparring partner agents.
     *
     * @return                   WEKA Instances object.
     */
    static private Instances generateTrainingData( Battle battle, int numTrainingGames, int numStepsInGame,
                                                   int msPerMove, Agent agent, Agent[] agentsSparring )
    {
        Random random = new Random();
        Instances instances = createInstances( battle.getDeck() );
        double[] values = new double[instances.numAttributes()];
        double [] scoreTotal = new double[2];
        double [] score = new double[2];


        Agent[] agents = new Agent[2];

        // Run training games.
        System.out.println( "Training games: " + numTrainingGames );
        for ( int a=0; a<score.length; a++ ) {
            scoreTotal[a] = 0.0;
        }

        GameLog log = new GameLog();
        for ( int n=0; n < numTrainingGames; ++n ) {

            int indexA = random.nextInt(2);
            int indexO = ((indexA==0) ? 1 : 0 );

            agents[indexA] = agent;
            agents[indexO] = agentsSparring[random.nextInt(agentsSparring.length)]; // pick sparring partner at random.

            // Run a game.
            log.clear();
            battle.run( false, numStepsInGame, msPerMove, agents, score, log );
            scoreTotal[0] += score[indexA];
            scoreTotal[1] += score[indexO];

            // Create Weka instance data from game.
            //  - Note that we need to get the action played in a state, from the subsequent game log record.
            boolean firstPass = true;
            StateAgent a = null, o = null;
            for ( StateBattle bs : log.getLog() ) {
                //System.out.println( bs.toString() );
                if ( firstPass ) {
                    firstPass = false;
                }
                else {
                    values[0] = a.getCol();
                    values[1] = a.getRow();
                    values[2] = a.getHealthPoints();
                    values[3] = a.getStaminaPoints();
                    values[4] = o.getCol();
                    values[5] = o.getRow();
                    values[6] = o.getHealthPoints();
                    values[7] = o.getStaminaPoints();
                    values[8] = instances.attribute(8).indexOfValue( bs.getLastMoves()[indexA].getName() ); // move of agent.
                    instances.add( new Instance( 1.0, values.clone() ) );
                }
                a = bs.getAgentState(indexA);
                o = bs.getAgentState(indexO);
            }

        }
        for (double aScoreTotal : scoreTotal) {
            System.out.print(' ');
            System.out.print(aScoreTotal);
        }
        System.out.println();

        return instances;
    }


    /**
     *  This function creates the structure of the Instances, that is, the attributes and their type.
     *
     * @param  deck  The deck of cards to use (determines the possible values of the class attribute).
     *
     * @return       WEKA Instances object.
     */
    static private Instances createInstances( CardDeck deck )
    {
        FastVector attributes = new FastVector(); // Attributes
        FastVector actions = new FastVector();    // Class

        // Domain of class are all possible cards.
        for ( Card c : deck.getCards() ) {
            actions.addElement( c.getName() );
        }

        // Add the predicting attributes (all numeric)
        attributes.addElement(new Attribute("a_x"));
        attributes.addElement(new Attribute("a_y"));
        attributes.addElement(new Attribute("a_health"));
        attributes.addElement(new Attribute("a_stamina"));
        attributes.addElement(new Attribute("o_x"));
        attributes.addElement(new Attribute("o_y"));
        attributes.addElement(new Attribute("o_health"));
        attributes.addElement(new Attribute("o_stamina"));
        // Add the class, the action the a_ agent took in the given state (nominal).
        attributes.addElement(new Attribute("a_action", actions));

        Instances instances = new Instances( "AgentBattleHistory", attributes, 0 );
        instances.setClassIndex(8);
		return instances;
    }
}
