package EntIndex;
import Common.Constant;

import java.io.*;
import java.util.*;


/**
 * Created by ethan on 16/3/23.
 */
public class IndexBuilder {
	Map<String, String> labels = null;
    private List<AhoCorasickDoubleArrayTrie<String>> l_acdat=new ArrayList<AhoCorasickDoubleArrayTrie<String>>();
    public Constant cons = null;
    int label_num = 0;
    int index_num = 0;
    
    public void initCons(String[] args){
    	this.cons = new Constant();
    	cons.parseArgs(args);
    }
    
    //读入entity列表
    public int loadLabels()throws IOException {
    	if(this.cons == null){
    		System.out.println("please init constants first!");
    		return -1;
    	}
        BufferedReader reader = new BufferedReader(new FileReader(cons.label_list_path));
        String line;
        String[] items=null;
        this.labels = new HashMap<String, String>();
        while ((line = reader.readLine()) != null)
        {
            //add main entity
            items = line.split(cons.interval,2);
            if(items.length<2 || items[0].equals(""))
                continue;
            if(labels.containsKey(items[0]))
            	labels.put(items[0], labels.get(items[0])+cons.interval+items[1]);
            else
            	labels.put(items[0], items[1]);
        }
        reader.close();
        this.label_num = labels.size();
        return this.label_num;
    }

    //建立并保存entity list的索引,
    public void build()throws IOException{
    	if(this.cons == null){
        	System.out.println("please init cons first!");
        	return;
        }
        if(this.labels==null || this.label_num==0){
        	this.loadLabels();
        }
        int label_count = 0;
        AhoCorasickDoubleArrayTrie<String> tmp_acdat = null;
        Map<String, String> tmp_label_map = new TreeMap<>();
        boolean isMatchWords = false;
		if(cons.isMatchWords==1)
			isMatchWords = true;
        for (Map.Entry<String, String> entry : this.labels.entrySet()) {
        	tmp_label_map.put(entry.getKey(), entry.getValue());
        	label_count++;
        	if(label_count%cons.max_index_num==0 || label_count >= this.label_num){
        		tmp_acdat = new AhoCorasickDoubleArrayTrie<String>();
        		tmp_acdat.build(tmp_label_map);
        		tmp_acdat.setMatchUnit(isMatchWords);
        		this.l_acdat.add(tmp_acdat);
        		tmp_label_map.clear();
        		this.index_num++;
        	}
        }
    }
    
    public void save() throws Exception{
    	if(cons.trie_path!=null)
	    	for(int i=1;i<=this.index_num;i++){
	    		ObjectOutputStream out = new ObjectOutputStream(new FileOutputStream(cons.trie_path+String.valueOf(i)));
	    		this.l_acdat.get(i-1).save(out);
	    		out.close();
	    	}
    }
    
    public void load() throws Exception{
    	if(cons.trie_path!=null){
    		boolean isMatchWords = false;
    		if(cons.isMatchWords==1)
    			isMatchWords = true;
    		AhoCorasickDoubleArrayTrie<String> tmp_acdat = null;
    		String tmp_path = null;
    		int i = 1;
    		while(true){
	    		tmp_path = cons.trie_path+String.valueOf(i);
	    		File fin = new File(tmp_path);
	    		if(!fin.exists()) break;
	    		
    			ObjectInputStream oin = new ObjectInputStream(new FileInputStream(tmp_path));
    			tmp_acdat = new AhoCorasickDoubleArrayTrie<String>();
    			tmp_acdat.load(oin);
    			tmp_acdat.setMatchUnit(isMatchWords);
    			this.l_acdat.add(tmp_acdat);
    			oin.close();
    			i++;
    		}
    		this.index_num = i-1;
    		for(i=0;i<this.index_num;i++)
    			this.label_num += this.l_acdat.get(i).size();
    		System.out.printf("successfully load %d labels within %d indexes!", this.label_num, this.index_num);
    	}
   }
    
    public List<AhoCorasickDoubleArrayTrie<String>.Hit<String>> parseText(String doc){
    	if(this.l_acdat==null || this.index_num <1){
    		System.out.println("please build the indexes first!");
    		return null;
    	}
    	List<AhoCorasickDoubleArrayTrie<String>.Hit<String>> re_mentions = null;
    	re_mentions = this.l_acdat.get(0).parseText(doc);
    	for(int i=1;i<this.index_num;i++)
    		re_mentions.addAll(this.l_acdat.get(i).parseText(doc));
    	return re_mentions;
    }
    

}
