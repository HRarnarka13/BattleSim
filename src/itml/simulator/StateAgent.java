package itml.simulator;

/**
 *
 *  This class implements the type StateAgent, which keeps track of the state of an agent
 *  (i.e., location, stamina, and health).
 *
 * @author      Yngvi Bjornsson
 *
 * @version     %I%, %G%
 *
 */
public class StateAgent {

     // Maximum stamina level an agent can have.
     public static int MAX_STAMINA = 15;
     // Maximum health level an agent can have.
     public static int MAX_HEALTH  = 10;

     private int m_col;
     private int m_row;
     private int m_staminaPoints;
     private int m_healthPoints;

    /**
     *
     * Constructor, create a new state.
     *
     * @param  col            column location of agent
     * @param  row            row location of agent
     * @param  staminaPoints  initial stamina points of agent
     * @param  healthPoints   initial health points of agent.
     *
     */
    public StateAgent( int col, int row, int staminaPoints, int healthPoints ) {
         m_col = col;
         m_row = row;
         m_staminaPoints = Math.max(0, Math.min(staminaPoints, MAX_STAMINA) );
         m_healthPoints  = Math.max(0, Math.min(healthPoints, MAX_HEALTH) );
    }

    /**
      *
      * Copy Constructor
      *
      * @param  stateAgent     Agent state to copy.
      *
      */
    public StateAgent( StateAgent stateAgent ) {
        m_col = stateAgent.getCol();
        m_row = stateAgent.getRow();
        m_staminaPoints = stateAgent.getStaminaPoints();
        m_healthPoints = stateAgent.getHealthPoints();
    }


    /**
      *
      * Clone agent state.
      *
      * @return  new cloned instance.
      *
      */
    public Object clone() {
         return new StateAgent( this );
    }


    /**
      *
      * Get column location of agent.
      *
      * @return  An integer representing column location.
      *
      */
    public int getCol() {
         return m_col;
    }

    /**
      *
      * Get row location of agent.
      *
      * @return  An integer representing row location.
      *
      */
    public int getRow() {
         return m_row;
    }

    /**
      *
      * Get stamina points of agent.
      *
      * @return  An integer indicating stamina point level.
      *
      */
    public int getStaminaPoints() {
        return m_staminaPoints;
    }

    /**
      *
      * Get health points of agent.
      *
      * @return  An integer indicating health point level.
      *
      */
    public int getHealthPoints() {
        return m_healthPoints;
    }

    /**
      *
      * Set column location of agent.
      *
      * @param col  An integer indicating column location.
      *
      */
    public void setCol( int col ) {
        m_col = col;
    }

    /**
      *
      * Set row location of agent.
      *
      * @param row  An integer indicating row location.
      *
      */
    public void setRow( int row ) {
        m_row = row;
    }

    /**
      *
      * Set stamina points of agent (range 0...MAX_STAMINA).
      *
      * @param points  An integer specifying stamina points of agent.
      *
      */
    public void setStaminaPoints( int points ) {
        m_staminaPoints = Math.max(0, Math.min(MAX_STAMINA, points));

    }

    /**
      *
      * Set health points of agent (range 0...MAX_HEALTH).
      *
      * @param points  An integer specifying health points of agent.
      *
      */
    public void setHealthPoints( int points ) {
        m_healthPoints = Math.max(0, Math.min(MAX_HEALTH, points));
    }

 }

