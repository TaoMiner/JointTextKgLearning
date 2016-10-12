package EntIndex;
import Common.Constant;

import java.io.*;
import java.util.*;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import model.Entity;
import model.Mention;


/**
 * Created by ethan on 16/3/23.
 */
public class IndexBuilder {
    private AhoCorasickDoubleArrayTrie<String> acdat;
    public String doc = null;

    //读入entity列表
    public int loadEntity(String input_path)throws IOException {
        BufferedReader reader = new BufferedReader(new FileReader(input_path));
        String line;
        Constant.entities = new HashMap();
        String[] items=null;
        String[] tmp_items = null;
        while ((line = reader.readLine()) != null)
        {
            //add main entity
            items = line.split("::=",2);
            if(items.length<2)
                continue;
            Constant.entities.put(items[0], items[1]);
        }
        reader.close();
        return Constant.entities.size();
    }

    //建立并保存entity list的索引,
    public void build(String input_path)throws IOException{
        if(Constant.entities==null)
            System.out.println("#Entity：" + this.loadEntity(input_path));

        this.acdat = new AhoCorasickDoubleArrayTrie<String>();
        this.acdat.build(Constant.entities);
    }
    
    public void save(String trie_path) throws Exception{
    	 ObjectOutputStream out = new ObjectOutputStream(new FileOutputStream(trie_path));
    	 this.acdat.save(out);
    	 out.close();
    }
    
    public void load(String trie_path) throws Exception{
    	ObjectInputStream oin = new ObjectInputStream(new FileInputStream(trie_path));
    	if(this.acdat==null)
    		this.acdat = new AhoCorasickDoubleArrayTrie<String>();
    	this.acdat.load(oin);
    	oin.close();
   }
    
    public List<AhoCorasickDoubleArrayTrie<String>.Hit<String>> parseText(String doc){
    	this.doc = doc;
    	return this.acdat.parseText(doc);
    }
    
    public static List<Integer> findMention(List<AhoCorasickDoubleArrayTrie<String>.Hit<String>> mentionList, int doc_lenth, String doc){
    	int[] sent = new int[doc_lenth];
    	List<Integer> re_mentions = new ArrayList<>();
    	for(int i=0;i<doc_lenth;i++)
    		sent[i] = -1;
    	AhoCorasickDoubleArrayTrie<String>.Hit<String> tmp_hit = null;
    	for(int i=0;i<mentionList.size();i++){
    		tmp_hit = mentionList.get(i);
    		//if the mention's (begin-1) and (end+1) are not white space or the sentence begin or end, ignore it
//    		if(tmp_hit.begin-1>=0 && doc.charAt(tmp_hit.begin-1)!=' ') continue;
//    		if(tmp_hit.end<doc_lenth && doc.charAt(tmp_hit.end)!=' ') continue;
    		int last_begin_index = sent[tmp_hit.begin];
    		int last_end_index = sent[tmp_hit.end-1];
    		//if both equal -1, then update; or equal other number, then do nothing
    		if(last_begin_index == -1 && last_end_index == -1){
    			for(int j=tmp_hit.begin;j<tmp_hit.end;j++)
    				sent[j] = i;
				continue;
    		}
    		else if(last_begin_index!=last_end_index){
    			int length_begin = last_begin_index==-1 ? 0:(mentionList.get(last_begin_index).end - mentionList.get(last_begin_index).begin);
    			int length_end = last_end_index==-1 ? 0:(mentionList.get(last_end_index).end - mentionList.get(last_end_index).begin);
    			int length_cur = tmp_hit.end - tmp_hit.begin;
    			if(length_cur>length_begin && length_cur>length_end){
    				for(int j=tmp_hit.begin;j<tmp_hit.end;j++)
        				sent[j] = i;
    				if(length_begin!=0)
    					for(int j=mentionList.get(last_begin_index).begin;j<tmp_hit.begin;j++)
            				sent[j] = -1;
    				if(length_end!=0)
    					for(int j=tmp_hit.end;j<mentionList.get(last_end_index).end;j++)
            				sent[j] = -1;
    			}
    		}
        }
    	int cur_index = -1;
    	for(int i=0;i<doc_lenth;i++){
    		if(sent[i]!=-1 && sent[i]!=cur_index){
    			re_mentions.add(sent[i]);
    			cur_index = sent[i];
    		}
    	}
    	return re_mentions;
    }
    
    //利用sorted entity list, 调用doubleArrayTrie建立索引并保存
    //读入sorted entity list和索引, 使用findMention方法找到所有mention
    public static void main(String[] args) throws Exception
    {
        IndexBuilder ibd = new IndexBuilder();
//        ibd.build(Constant.entity_dic_sorted);
//        System.out.println("build trie finished! saving...");
//        ibd.save(Constant.entity_trie);
        System.out.println("loading trie...");
        long start = System.currentTimeMillis();
        ibd.load(Constant.entity_trie);
        long end = System.currentTimeMillis();
        System.out.println("load finished! times: "+ (end-start));
        BufferedReader reader = new BufferedReader(new FileReader(Constant.news_path));
    	String line = null;
    	String label = null;
    	StringBuffer sb = new StringBuffer();
    	List<AhoCorasickDoubleArrayTrie<String>.Hit<String>> can_mention_list = null;
    	Pattern p=Pattern.compile("[[(.*?)]]");
    	Matcher m=null;
    	List<Integer> mention_list = null;
    	int start_pos = 0;
    	int end_pos = 0;
    	HashMap<String, Integer> mention_count = new HashMap<>();
    	start = System.currentTimeMillis();
    	while((line = reader.readLine())!=null){
    		System.out.println(line);
    		m = p.matcher(line);
    		end_pos = line.length();
    		while(m.find()){
    			end_pos = m.start();
    			can_mention_list = ibd.parseText(line.substring(start_pos, end_pos));
    			mention_list = findMention(can_mention_list, end_pos-start_pos, line);
    			for(Integer index:mention_list){
    				label = line.substring(can_mention_list.get(index).begin, can_mention_list.get(index).end);
    				if(Constant.entities.containsKey(label)){
    					if(mention_count.containsKey(label))
    						mention_count.put(label, mention_count.get(label)+1);
    					else
    						mention_count.put(label, 1);
    				}
    			}
    			//add text mention count
    			start_pos = m.end();
    		}
    		if(start_pos!=end_pos){
    			can_mention_list = ibd.parseText(line.substring(start_pos, end_pos));
    			mention_list = findMention(can_mention_list, end_pos-start_pos, line);
    			for(Integer index:mention_list){
    				label = line.substring(can_mention_list.get(index).begin, can_mention_list.get(index).end);

					if(mention_count.containsKey(label))
						mention_count.put(label, mention_count.get(label)+1);
					else
						mention_count.put(label, 1);
    			}
    		}
    		label = null;
    		can_mention_list = null;
    		mention_list = null;
    		m = null;
    		line = null;
    	}
        
        end = System.currentTimeMillis();
        System.out.println("time: " + (start - end));
        for (Map.Entry<String, Integer> entry : mention_count.entrySet()) {
            System.out.println(entry.getKey() + " = " + entry.getValue());
        } 
    }
}
